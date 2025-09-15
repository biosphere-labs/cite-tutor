"""
Training abstraction layer for multi-stage academic model training.
Handles both cloud and local environments with budget constraints under $10.
Supports multiple academic domains through configuration.
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    TrainerCallback, TrainerState, TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import time
import psutil
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import boto3
import requests
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for individual training stage."""
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    description: str


@dataclass
class EnvironmentConfig:
    """Configuration for training environment."""
    name: str
    base_model: str
    quantization: str
    stages: Dict[int, StageConfig]
    estimated_total_hours: float
    estimated_cost_usd: float = 0.0
    auto_shutdown: bool = False
    instance_type: str = "local"


class CostLimitCallback(TrainerCallback):
    """Monitor training cost and enforce budget limits."""

    def __init__(self, budget_limit: float, start_time: float, hourly_rate: float = 0.526):
        self.budget_limit = budget_limit
        self.start_time = start_time
        self.hourly_rate = hourly_rate
        self.last_cost_check = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Check cost at each training step."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        current_cost = elapsed_hours * self.hourly_rate

        # Stop if approaching budget limit
        if current_cost >= self.budget_limit * 0.95:
            logger.warning(f"Approaching budget limit (${current_cost:.2f}/${self.budget_limit:.2f}), stopping training")
            control.should_training_stop = True
            return control

        # Log cost periodically
        if state.global_step % 100 == 0 and state.global_step != self.last_cost_check:
            logger.info(f"Training cost: ${current_cost:.2f} / ${self.budget_limit:.2f} (Step {state.global_step})")
            self.last_cost_check = state.global_step

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log cost at end of each epoch."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        current_cost = elapsed_hours * self.hourly_rate
        logger.info(f"Epoch {state.epoch} complete. Total cost so far: ${current_cost:.2f}")


class TrainingResourceManager:
    """Manages training resources across different environments with budget control."""

    # Multi-stage configurations for different environments
    MULTI_STAGE_CONFIGS = {
        "local_4gb": EnvironmentConfig(
            name="local_4gb",
            base_model="distilgpt2",
            quantization="4bit",
            stages={
                1: StageConfig(epochs=2, batch_size=1, grad_accum=32, lr=1e-4, description="Book foundation"),
                2: StageConfig(epochs=2, batch_size=1, grad_accum=32, lr=5e-5, description="Paper integration"),
                3: StageConfig(epochs=1, batch_size=1, grad_accum=32, lr=2e-5, description="Knowledge synthesis")
            },
            estimated_total_hours=12.0,
            estimated_cost_usd=0.0,  # Local training
            auto_shutdown=False,
            instance_type="local"
        ),

        "local_8gb": EnvironmentConfig(
            name="local_8gb",
            base_model="microsoft/DialoGPT-small",
            quantization="4bit",
            stages={
                1: StageConfig(epochs=3, batch_size=2, grad_accum=16, lr=1.5e-4, description="Book foundation"),
                2: StageConfig(epochs=2, batch_size=2, grad_accum=16, lr=7e-5, description="Paper integration"),
                3: StageConfig(epochs=1, batch_size=2, grad_accum=16, lr=3e-5, description="Knowledge synthesis")
            },
            estimated_total_hours=8.0,
            estimated_cost_usd=0.0,
            auto_shutdown=False,
            instance_type="local"
        ),

        "cloud_aws_budget": EnvironmentConfig(
            name="cloud_aws_budget",
            base_model="microsoft/DialoGPT-medium",
            quantization="8bit",
            stages={
                1: StageConfig(epochs=3, batch_size=4, grad_accum=8, lr=2e-4, description="Book foundation"),
                2: StageConfig(epochs=2, batch_size=4, grad_accum=8, lr=1e-4, description="Paper integration"),
                3: StageConfig(epochs=1, batch_size=4, grad_accum=8, lr=5e-5, description="Knowledge synthesis")
            },
            estimated_total_hours=3.5,
            estimated_cost_usd=1.84,  # $0.526/hour * 3.5 hours
            auto_shutdown=True,
            instance_type="ml.g4dn.xlarge"
        ),

        "cloud_gcp_budget": EnvironmentConfig(
            name="cloud_gcp_budget",
            base_model="microsoft/DialoGPT-medium",
            quantization="8bit",
            stages={
                1: StageConfig(epochs=3, batch_size=4, grad_accum=8, lr=2e-4, description="Book foundation"),
                2: StageConfig(epochs=2, batch_size=4, grad_accum=8, lr=1e-4, description="Paper integration"),
                3: StageConfig(epochs=1, batch_size=4, grad_accum=8, lr=5e-5, description="Knowledge synthesis")
            },
            estimated_total_hours=3.5,
            estimated_cost_usd=2.10,  # $0.60/hour * 3.5 hours (n1-standard-4 + T4)
            auto_shutdown=True,
            instance_type="n1-standard-4"
        ),

        "cloud_colab_pro": EnvironmentConfig(
            name="cloud_colab_pro",
            base_model="distilgpt2",
            quantization="4bit",
            stages={
                1: StageConfig(epochs=2, batch_size=2, grad_accum=16, lr=1e-4, description="Book foundation"),
                2: StageConfig(epochs=2, batch_size=2, grad_accum=16, lr=5e-5, description="Paper integration"),
                3: StageConfig(epochs=1, batch_size=2, grad_accum=16, lr=2e-5, description="Knowledge synthesis")
            },
            estimated_total_hours=6.0,
            estimated_cost_usd=0.0,  # Colab Pro included
            auto_shutdown=False,
            instance_type="colab_pro"
        )
    }

    def __init__(self, budget_limit: float = 10.0):
        self.budget_limit = budget_limit
        self.environment = self.detect_environment()
        self.config = self.get_multi_stage_config()

    def detect_environment(self) -> str:
        """Detect the current training environment."""

        try:
            # Check if running in Google Colab
            import google.colab
            return "cloud_colab_pro"
        except ImportError:
            pass

        # Check if running on AWS
        try:
            response = requests.get('http://169.254.169.254/latest/meta-data/', timeout=2)
            if response.status_code == 200:
                return "cloud_aws_budget"
        except:
            pass

        # Check if running on GCP
        try:
            response = requests.get('http://metadata.google.internal/computeMetadata/v1/',
                                   headers={'Metadata-Flavor': 'Google'}, timeout=2)
            if response.status_code == 200:
                return "cloud_gcp_budget"
        except:
            pass

        # Local environment - detect GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb > 7:
                return "local_8gb"
            else:
                return "local_4gb"
        else:
            return "local_4gb"  # CPU fallback

    def get_multi_stage_config(self, environment: str = None, budget_limit: float = None) -> EnvironmentConfig:
        """Get multi-stage configuration for environment and budget."""

        env = environment or self.environment
        budget = budget_limit or self.budget_limit

        if env not in self.MULTI_STAGE_CONFIGS:
            logger.warning(f"Unknown environment {env}, using local_4gb")
            env = "local_4gb"

        config = self.MULTI_STAGE_CONFIGS[env]

        # Check if config fits within budget
        if config.estimated_cost_usd > budget:
            logger.warning(f"Config cost ${config.estimated_cost_usd:.2f} exceeds budget ${budget:.2f}")
            # Fall back to cheaper local option
            config = self.MULTI_STAGE_CONFIGS["local_4gb"]

        logger.info(f"Selected configuration: {config.name}")
        logger.info(f"Estimated cost: ${config.estimated_cost_usd:.2f}")
        logger.info(f"Estimated time: {config.estimated_total_hours:.1f} hours")

        return config

    def setup_stage_model(self, stage_num: int, previous_stage_path: str = None) -> Tuple[Any, Any]:
        """Setup model for specific training stage."""

        config = self.config

        if previous_stage_path and Path(previous_stage_path).exists():
            # Load from previous stage
            logger.info(f"Loading model from previous stage: {previous_stage_path}")
            model = AutoModelForCausalLM.from_pretrained(previous_stage_path)
            tokenizer = AutoTokenizer.from_pretrained(previous_stage_path)

        else:
            # Setup fresh model
            logger.info(f"Setting up fresh model: {config.base_model}")
            model, tokenizer = self._setup_optimized_model(config)

        return model, tokenizer

    def _setup_optimized_model(self, config: EnvironmentConfig) -> Tuple[Any, Any]:
        """Setup model with environment-specific optimizations."""

        model_name = config.base_model

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Setup quantization based on config
        if config.quantization == "4bit" and config.name.startswith("local"):
            # 4-bit quantization for local 4GB VRAM
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                model = prepare_model_for_kbit_training(model)

            except ImportError:
                logger.warning("bitsandbytes not available, using float16")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

        elif config.quantization == "8bit":
            # 8-bit quantization for cloud environments
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True
                )
                model = prepare_model_for_kbit_training(model)

            except ImportError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

        else:
            # Standard loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Apply LoRA for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            r=8 if config.name.startswith("local") else 16,  # Lower rank for local
            lora_alpha=16 if config.name.startswith("local") else 32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self._get_target_modules(model_name)
        )

        model = get_peft_model(model, lora_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model, tokenizer

    def _get_target_modules(self, model_name: str) -> List[str]:
        """Get target modules for LoRA based on model architecture."""
        if 'gpt2' in model_name.lower() or 'dialogpt' in model_name.lower():
            return ["c_attn", "c_proj"]
        elif 'llama' in model_name.lower():
            return ["q_proj", "v_proj"]
        elif 't5' in model_name.lower():
            return ["q", "v"]
        else:
            return ["q_proj", "v_proj"]  # Default

    def estimate_multi_stage_cost(self, datasets: List[Dataset]) -> float:
        """Estimate total cost for multi-stage training."""

        config = self.config

        if config.estimated_cost_usd == 0.0:  # Local training
            return 0.0

        # Estimate based on dataset sizes and configuration
        total_samples = sum(len(dataset) for dataset in datasets)

        # Estimate training steps per stage
        total_steps = 0
        for stage_num, stage_config in config.stages.items():
            dataset_idx = min(stage_num - 1, len(datasets) - 1)
            dataset_size = len(datasets[dataset_idx]) if datasets else 1000

            steps_per_epoch = dataset_size // (stage_config.batch_size * stage_config.grad_accum)
            stage_steps = steps_per_epoch * stage_config.epochs
            total_steps += stage_steps

        # Estimate time based on steps (rough approximation)
        estimated_hours = total_steps * 0.001  # ~1000 steps per hour (very rough)
        estimated_hours = max(estimated_hours, config.estimated_total_hours * 0.5)  # Minimum estimate
        estimated_hours = min(estimated_hours, config.estimated_total_hours * 2.0)  # Maximum estimate

        # Calculate cost based on instance type
        hourly_rates = {
            "ml.g4dn.xlarge": 0.526,  # AWS
            "n1-standard-4": 0.60,    # GCP with T4
            "colab_pro": 0.0,
            "local": 0.0
        }

        hourly_rate = hourly_rates.get(config.instance_type, 0.0)
        estimated_cost = estimated_hours * hourly_rate

        logger.info(f"Cost estimation: {estimated_hours:.1f} hours * ${hourly_rate:.3f}/hour = ${estimated_cost:.2f}")

        return estimated_cost


class UniversalMultiStageTrainer:
    """Universal trainer that handles multi-stage training across environments."""

    def __init__(self, budget_limit: float = 10.0, output_dir: str = "outputs/training"):
        self.resource_manager = TrainingResourceManager(budget_limit)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = None

    def train_academic_model_multistage(self,
                                       book_data: Dataset,
                                       paper_data: Dataset,
                                       integrated_data: Dataset) -> Dict:
        """Execute 3-stage training with cost/memory optimization."""

        config = self.resource_manager.config
        datasets = [book_data, paper_data, integrated_data]

        # Cost estimation and validation
        total_cost = self.resource_manager.estimate_multi_stage_cost(datasets)
        if total_cost > self.resource_manager.budget_limit:
            raise Exception(f"Estimated cost ${total_cost:.2f} exceeds budget ${self.resource_manager.budget_limit:.2f}")

        logger.info("=" * 60)
        logger.info("MULTI-STAGE ACADEMIC MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Environment: {config.name}")
        logger.info(f"Base model: {config.base_model}")
        logger.info(f"Quantization: {config.quantization}")
        logger.info(f"Estimated cost: ${total_cost:.2f}")
        logger.info(f"Estimated time: {config.estimated_total_hours:.1f} hours")
        logger.info(f"Budget limit: ${self.resource_manager.budget_limit:.2f}")
        logger.info("=" * 60)

        self.start_time = time.time()
        results = {
            'config': {
                'environment': config.name,
                'base_model': config.base_model,
                'estimated_cost': total_cost,
                'budget_limit': self.resource_manager.budget_limit
            },
            'stages': {}
        }

        # Stage 1: Book knowledge foundation
        logger.info("🔬 STAGE 1: Book Knowledge Foundation Training")
        stage1_result = self.execute_single_stage(
            stage_num=1,
            data=book_data,
            config=config,
            output_path=str(self.output_dir / "stage1"),
            previous_stage_path=None
        )
        results['stages']['stage1'] = stage1_result

        # Stage 2: Foundational paper integration
        logger.info("📚 STAGE 2: Foundational Paper Integration")
        stage2_result = self.execute_single_stage(
            stage_num=2,
            data=paper_data,
            config=config,
            output_path=str(self.output_dir / "stage2"),
            previous_stage_path=str(self.output_dir / "stage1")
        )
        results['stages']['stage2'] = stage2_result

        # Stage 3: Knowledge synthesis
        logger.info("🧠 STAGE 3: Knowledge Synthesis")
        stage3_result = self.execute_single_stage(
            stage_num=3,
            data=integrated_data,
            config=config,
            output_path=str(self.output_dir / "final"),
            previous_stage_path=str(self.output_dir / "stage2")
        )
        results['stages']['stage3'] = stage3_result

        # Calculate final costs and metrics
        total_time = time.time() - self.start_time
        actual_cost = self.calculate_actual_cost(total_time)

        results['summary'] = {
            'total_time_hours': total_time / 3600,
            'actual_cost_usd': actual_cost,
            'cost_under_budget': actual_cost <= self.resource_manager.budget_limit,
            'final_model_path': str(self.output_dir / "final")
        }

        logger.info("=" * 60)
        logger.info("MULTI-STAGE TRAINING COMPLETE!")
        logger.info(f"✅ Total time: {total_time/3600:.1f} hours")
        logger.info(f"✅ Actual cost: ${actual_cost:.2f}")
        logger.info(f"✅ Final model: {self.output_dir / 'final'}")
        logger.info("=" * 60)

        # Save results
        self.save_training_results(results)

        return results

    def execute_single_stage(self, stage_num: int, data: Dataset, config: EnvironmentConfig,
                           output_path: str, previous_stage_path: str = None) -> Dict:
        """Execute single training stage with environment optimization."""

        stage_config = config.stages[stage_num]
        stage_output_dir = Path(output_path)
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Stage {stage_num}: {stage_config.description}")
        logger.info(f"Epochs: {stage_config.epochs}, Batch size: {stage_config.batch_size}, LR: {stage_config.lr}")

        # Setup model for this stage
        model, tokenizer = self.resource_manager.setup_stage_model(stage_num, previous_stage_path)

        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=str(stage_output_dir),

            # Core training parameters
            per_device_train_batch_size=stage_config.batch_size,
            gradient_accumulation_steps=stage_config.grad_accum,
            learning_rate=stage_config.lr,
            num_train_epochs=stage_config.epochs,

            # Memory and precision optimization
            fp16=True,
            gradient_checkpointing=True,
            dataloader_drop_last=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,

            # Optimizer selection based on environment
            optim="adamw_bnb_8bit" if config.quantization in ["4bit", "8bit"] else "adamw_hf",

            # Evaluation and saving
            eval_strategy="steps" if len(data) > 1000 else "no",
            eval_steps=max(100, len(data) // (stage_config.batch_size * stage_config.grad_accum) // 4),
            save_steps=max(200, len(data) // (stage_config.batch_size * stage_config.grad_accum) // 2),
            save_total_limit=1,  # Save space
            load_best_model_at_end=True if len(data) > 1000 else False,

            # Logging
            logging_steps=25,
            logging_dir=str(stage_output_dir / "logs"),
            report_to=None,

            # Regularization
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_ratio=0.1 if stage_num == 1 else 0.05,  # Less warmup for later stages

            # Memory management
            prediction_loss_only=True,
            include_inputs_for_metrics=False,

            # Reproducibility
            seed=42,
            data_seed=42,
        )

        # Setup callbacks
        callbacks = []

        # Add cost monitoring for cloud environments
        if config.estimated_cost_usd > 0.0 and self.start_time:
            hourly_rate = config.estimated_cost_usd / config.estimated_total_hours
            callbacks.append(CostLimitCallback(self.resource_manager.budget_limit, self.start_time, hourly_rate))

        # Prepare dataset for training
        formatted_data = self.format_dataset_for_training(data, tokenizer)

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=formatted_data,
            tokenizer=tokenizer,
            callbacks=callbacks
        )

        # Train model
        train_start = time.time()
        train_result = trainer.train()
        train_time = time.time() - train_start

        # Save model
        trainer.save_model(str(stage_output_dir))
        tokenizer.save_pretrained(str(stage_output_dir))

        # Clear memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stage_result = {
            'stage_num': stage_num,
            'description': stage_config.description,
            'training_loss': train_result.training_loss,
            'training_time_minutes': train_time / 60,
            'epochs_completed': stage_config.epochs,
            'output_path': str(stage_output_dir),
            'dataset_size': len(data)
        }

        logger.info(f"Stage {stage_num} complete: Loss {train_result.training_loss:.4f}, Time {train_time/60:.1f}min")

        return stage_result

    def format_dataset_for_training(self, dataset: Dataset, tokenizer: Any) -> Dataset:
        """Format dataset for causal language modeling."""

        def tokenize_function(examples):
            # Format as "Question: ... Answer: ..." if not already formatted
            texts = []
            for i in range(len(examples.get('question', examples.get('text', [''])))):
                if 'question' in examples:
                    question = examples['question'][i]
                    answer = examples['answer'][i] if 'answer' in examples else ""
                    text = f"Question: {question}\nAnswer: {answer}"
                else:
                    text = examples['text'][i]
                texts.append(text)

            # Tokenize
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )

            # Labels are same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        return tokenized_dataset

    def calculate_actual_cost(self, total_time_seconds: float) -> float:
        """Calculate actual training cost based on time and environment."""

        config = self.resource_manager.config

        if config.estimated_cost_usd == 0.0:  # Local training
            return 0.0

        # Calculate hourly rate
        hourly_rate = config.estimated_cost_usd / config.estimated_total_hours
        actual_hours = total_time_seconds / 3600
        actual_cost = actual_hours * hourly_rate

        return actual_cost

    def save_training_results(self, results: Dict):
        """Save training results to JSON file."""

        output_file = self.output_dir / "training_results.json"

        # Add metadata
        results['metadata'] = {
            'completion_time': datetime.now().isoformat(),
            'environment_detected': self.resource_manager.environment,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            results['metadata']['gpu_name'] = torch.cuda.get_device_name()
            results['metadata']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Training results saved to: {output_file}")


def create_sample_datasets(domain: str = None) -> Tuple[Dataset, Dataset, Dataset]:
    """Create sample datasets for testing using domain configuration."""

    from domain_config import get_domain_config

    domain_config = get_domain_config()
    if domain:
        domain_config.set_domain(domain)

    # Get sample data from domain configuration
    book_samples = domain_config.get_sample_data('book')
    paper_samples = domain_config.get_sample_data('paper')
    integrated_samples = domain_config.get_sample_data('integrated')

    # Create datasets from sample data
    book_data = Dataset.from_dict({
        'question': [item['question'] for item in book_samples] if book_samples else ['Sample book question?'],
        'answer': [item['answer'] for item in book_samples] if book_samples else ['Sample book answer.']
    })

    paper_data = Dataset.from_dict({
        'question': [item['question'] for item in paper_samples] if paper_samples else ['Sample paper question?'],
        'answer': [item['answer'] for item in paper_samples] if paper_samples else ['Sample paper answer.']
    })

    integrated_data = Dataset.from_dict({
        'question': [item['question'] for item in integrated_samples] if integrated_samples else ['Sample integrated question?'],
        'answer': [item['answer'] for item in integrated_samples] if integrated_samples else ['Sample integrated answer.']
    })

    return book_data, paper_data, integrated_data


if __name__ == "__main__":
    # Example usage
    trainer = UniversalMultiStageTrainer(budget_limit=10.0)

    logger.info(f"Detected environment: {trainer.resource_manager.environment}")
    logger.info(f"Selected configuration: {trainer.resource_manager.config.name}")

    # Create sample datasets
    book_data, paper_data, integrated_data = create_sample_datasets()

    # Estimate cost
    estimated_cost = trainer.resource_manager.estimate_multi_stage_cost([book_data, paper_data, integrated_data])
    logger.info(f"Estimated training cost: ${estimated_cost:.2f}")

    if estimated_cost <= trainer.resource_manager.budget_limit:
        logger.info("✅ Training is within budget!")
    else:
        logger.warning("❌ Training exceeds budget limit!")

    print("Universal Multi-Stage Trainer ready for chemistry model training!")