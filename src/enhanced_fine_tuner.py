"""
Enhanced fine-tuning controller for integrated book + paper knowledge.
Multi-stage approach optimized for 4GB VRAM with aggressive memory management.
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    TrainerCallback, TrainerState, TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import yaml
import time
import psutil
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryMonitorCallback(TrainerCallback):
    """Monitor GPU memory during training with 4GB VRAM constraints."""

    def __init__(self, max_memory_gb: float = 3.8):
        self.max_memory_gb = max_memory_gb
        self.memory_warnings = 0
        self.last_cleanup = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor memory at each training step."""
        if torch.cuda.is_available():
            memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            memory_cached_gb = torch.cuda.memory_reserved() / (1024**3)

            # Log memory usage periodically
            if state.global_step % 50 == 0:
                logger.info(f"Step {state.global_step}: GPU Memory - Allocated: {memory_used_gb:.2f}GB, Cached: {memory_cached_gb:.2f}GB")

            # Force cleanup if approaching limit
            if memory_used_gb > self.max_memory_gb:
                self.memory_warnings += 1
                logger.warning(f"High memory usage: {memory_used_gb:.2f}GB (Warning #{self.memory_warnings})")

                # Force garbage collection and cache clearing
                torch.cuda.empty_cache()
                import gc
                gc.collect()

                # Emergency stop if memory issues persist
                if self.memory_warnings > 10:
                    logger.error("Persistent memory issues, stopping training")
                    control.should_training_stop = True

    def on_epoch_end(self, args, state, control, **kwargs):
        """Clean up memory at epoch end."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleaned GPU cache at epoch end")


class KnowledgeRetentionCallback(TrainerCallback):
    """Monitor knowledge retention across training stages."""

    def __init__(self, eval_questions: List[Dict]):
        self.eval_questions = eval_questions
        self.retention_scores = []

    def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader, **kwargs):
        """Evaluate knowledge retention on key questions."""
        if len(self.eval_questions) == 0:
            return

        model.eval()
        retention_score = 0.0

        try:
            for question_data in self.eval_questions[:10]:  # Sample 10 questions
                question = question_data.get('question', '')
                expected_answer = question_data.get('answer', '')

                # Generate answer
                inputs = tokenizer.encode(question, return_tensors="pt", max_length=128, truncation=True)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs.to(model.device),
                        max_length=200,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Simple retention scoring (could be enhanced with BLEU/ROUGE)
                if expected_answer.lower() in generated_answer.lower():
                    retention_score += 1.0

            retention_score = retention_score / len(self.eval_questions[:10])
            self.retention_scores.append(retention_score)

            logger.info(f"Knowledge retention score: {retention_score:.2f}")

        except Exception as e:
            logger.warning(f"Knowledge retention evaluation failed: {e}")

        model.train()


class IntegratedFineTuner:
    """
    Enhanced fine-tuning controller for integrated book + paper knowledge.
    Uses multi-stage approach optimized for 4GB VRAM.
    """

    def __init__(self, config_path: str = "config/models.yaml", output_dir: str = "outputs/models"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training stage configurations
        self.stage_configs = {
            'stage1_books': {
                'description': 'Book knowledge foundation training',
                'epochs': 2,
                'learning_rate': 1e-4,
                'warmup_ratio': 0.1
            },
            'stage2_papers': {
                'description': 'Foundational paper integration',
                'epochs': 1,
                'learning_rate': 5e-5,  # Lower LR to preserve book knowledge
                'warmup_ratio': 0.05
            },
            'stage3_synthesis': {
                'description': 'Knowledge synthesis and integration',
                'epochs': 1,
                'learning_rate': 2e-5,  # Lowest LR for fine refinement
                'warmup_ratio': 0.03
            }
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration for 4GB VRAM fine-tuning."""
        return {
            'fine_tuning': {
                'base_model': 'distilgpt2',
                'quantization': '4bit',
                'lora_r': 8,
                'lora_alpha': 16,
                'batch_size': 1,
                'gradient_accumulation': 32
            },
            'memory_limits': {
                'max_gpu_memory_mb': 4096,
                'safety_buffer_mb': 512
            }
        }

    def setup_4gb_optimized_model(self) -> Tuple[Any, Any]:
        """Setup model with aggressive 4GB VRAM optimization."""

        try:
            import bitsandbytes as bnb
        except ImportError:
            logger.error("bitsandbytes not available. Install with: pip install bitsandbytes")
            raise

        model_name = self.config['fine_tuning']['base_model']
        logger.info(f"Setting up 4GB optimized model: {model_name}")

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Important for generation

        # Configure 4-bit quantization
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['fine_tuning']['lora_r'],
            lora_alpha=self.config['fine_tuning']['lora_alpha'],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self._get_target_modules(model_name)
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model, tokenizer

    def _get_target_modules(self, model_name: str) -> List[str]:
        """Get target modules for LoRA based on model architecture."""
        if 'gpt2' in model_name.lower():
            return ["c_attn", "c_proj"]
        elif 'llama' in model_name.lower():
            return ["q_proj", "v_proj"]
        elif 't5' in model_name.lower():
            return ["q", "v"]
        else:
            # Default for most transformer models
            return ["q_proj", "v_proj"]

    def get_4gb_training_args(self, stage_name: str) -> TrainingArguments:
        """Training arguments optimized for 4GB VRAM."""

        stage_config = self.stage_configs[stage_name]

        return TrainingArguments(
            output_dir=str(self.output_dir / stage_name),

            # Memory optimization
            per_device_train_batch_size=self.config['fine_tuning']['batch_size'],
            gradient_accumulation_steps=self.config['fine_tuning']['gradient_accumulation'],
            per_device_eval_batch_size=1,

            # Precision and memory management
            fp16=True,
            dataloader_drop_last=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Avoid multiprocessing overhead

            # Optimizer for low memory
            optim="adamw_bnb_8bit",

            # Learning parameters
            learning_rate=stage_config['learning_rate'],
            num_train_epochs=stage_config['epochs'],
            warmup_ratio=stage_config['warmup_ratio'],
            weight_decay=0.01,
            max_grad_norm=1.0,

            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,

            # Logging
            logging_steps=25,
            logging_dir=str(self.output_dir / stage_name / "logs"),
            report_to=None,  # Disable wandb to save memory

            # Memory management
            prediction_loss_only=True,
            include_inputs_for_metrics=False,

            # Disable features that use extra memory
            label_smoothing_factor=0.0,

            # Reproducibility
            seed=42,
            data_seed=42,
        )

    def prepare_training_data(self, qa_data: List[Dict]) -> Dataset:
        """Prepare Q&A data for training."""

        # Format data for causal language modeling
        training_texts = []

        for qa in qa_data:
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()

            if question and answer:
                # Format: "Question: ... Answer: ..."
                formatted_text = f"Question: {question}\nAnswer: {answer}"
                training_texts.append(formatted_text)

        logger.info(f"Prepared {len(training_texts)} training examples")

        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})

        return dataset

    def tokenize_dataset(self, dataset: Dataset, tokenizer: Any, max_length: int = 512) -> Dataset:
        """Tokenize dataset for training."""

        def tokenize_function(examples):
            # Tokenize texts
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        return tokenized_dataset

    def train_with_memory_monitoring(self, model: Any, tokenizer: Any, dataset: Dataset, stage_name: str, eval_questions: List[Dict] = None) -> Dict:
        """Training with 4GB VRAM monitoring and knowledge retention tracking."""

        logger.info(f"Starting {stage_name}: {self.stage_configs[stage_name]['description']}")

        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)

        # Split dataset (90% train, 10% eval)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        # Get training arguments
        training_args = self.get_4gb_training_args(stage_name)

        # Setup callbacks
        callbacks = [MemoryMonitorCallback(max_memory_gb=3.8)]

        if eval_questions:
            callbacks.append(KnowledgeRetentionCallback(eval_questions))

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks
        )

        # Log GPU memory before training
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Pre-training GPU memory: {memory_gb:.2f}GB")

        # Train model
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time

        # Log training results
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")

        # Save model checkpoint
        checkpoint_dir = self.output_dir / f"{stage_name}_checkpoint"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Clear memory
        torch.cuda.empty_cache()

        return {
            'training_loss': train_result.training_loss,
            'training_time': training_time,
            'global_step': train_result.global_step,
            'checkpoint_dir': str(checkpoint_dir)
        }

    def execute_multi_stage_training(self, book_data: List[Dict], paper_data: List[Dict], integrated_data: List[Dict]) -> Dict:
        """Execute complete multi-stage training pipeline."""

        logger.info("Starting multi-stage integrated fine-tuning")
        results = {}

        try:
            # Stage 1: Book knowledge foundation
            logger.info("=" * 60)
            logger.info("STAGE 1: Book Knowledge Foundation Training")
            logger.info("=" * 60)

            model, tokenizer = self.setup_4gb_optimized_model()
            book_dataset = self.prepare_training_data(book_data)

            stage1_results = self.train_with_memory_monitoring(
                model, tokenizer, book_dataset, "stage1_books", book_data[:20]  # Sample eval questions
            )
            results['stage1'] = stage1_results

            # Clear memory completely
            del model
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            # Stage 2: Paper knowledge integration
            logger.info("=" * 60)
            logger.info("STAGE 2: Foundational Paper Knowledge Integration")
            logger.info("=" * 60)

            # Load Stage 1 checkpoint
            checkpoint_dir = results['stage1']['checkpoint_dir']
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

            paper_dataset = self.prepare_training_data(paper_data)

            stage2_results = self.train_with_memory_monitoring(
                model, tokenizer, paper_dataset, "stage2_papers", paper_data[:20]
            )
            results['stage2'] = stage2_results

            # Clear memory
            del model
            torch.cuda.empty_cache()
            gc.collect()

            # Stage 3: Knowledge synthesis
            logger.info("=" * 60)
            logger.info("STAGE 3: Knowledge Synthesis and Integration")
            logger.info("=" * 60)

            # Load Stage 2 checkpoint
            checkpoint_dir = results['stage2']['checkpoint_dir']
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

            integrated_dataset = self.prepare_training_data(integrated_data)

            stage3_results = self.train_with_memory_monitoring(
                model, tokenizer, integrated_dataset, "stage3_synthesis", integrated_data[:20]
            )
            results['stage3'] = stage3_results

            # Final model save
            final_model_dir = self.output_dir / "final_model"
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"Final model saved to: {final_model_dir}")

            results['final_model_path'] = str(final_model_dir)

        except Exception as e:
            logger.error(f"Multi-stage training failed: {e}")
            raise

        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def evaluate_knowledge_integration(self, model_path: str, test_questions: List[Dict]) -> Dict:
        """Evaluate integrated knowledge across book and paper sources."""

        logger.info(f"Evaluating knowledge integration on {len(test_questions)} test questions")

        # Load final model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model.eval()

        evaluation_results = {
            'total_questions': len(test_questions),
            'book_knowledge_score': 0.0,
            'paper_knowledge_score': 0.0,
            'integration_score': 0.0,
            'detailed_results': []
        }

        book_correct = 0
        paper_correct = 0
        integration_correct = 0

        with torch.no_grad():
            for i, question_data in enumerate(test_questions):
                question = question_data.get('question', '')
                expected_answer = question_data.get('answer', '')
                knowledge_type = question_data.get('knowledge_type', 'unknown')

                # Generate answer
                inputs = tokenizer.encode(f"Question: {question}\nAnswer:", return_tensors="pt", max_length=256, truncation=True)

                outputs = model.generate(
                    inputs.to(model.device),
                    max_length=400,
                    do_sample=False,
                    num_beams=2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_answer = generated_text.split("Answer:")[-1].strip()

                # Simple evaluation (could be enhanced with semantic similarity)
                is_correct = self._evaluate_answer_quality(expected_answer, generated_answer)

                # Track by knowledge type
                if 'book' in knowledge_type:
                    book_correct += is_correct
                elif 'paper' in knowledge_type:
                    paper_correct += is_correct
                elif 'integration' in knowledge_type:
                    integration_correct += is_correct

                evaluation_results['detailed_results'].append({
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': generated_answer,
                    'knowledge_type': knowledge_type,
                    'correct': is_correct
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(test_questions)} questions")

        # Calculate scores
        book_questions = sum(1 for q in test_questions if 'book' in q.get('knowledge_type', ''))
        paper_questions = sum(1 for q in test_questions if 'paper' in q.get('knowledge_type', ''))
        integration_questions = sum(1 for q in test_questions if 'integration' in q.get('knowledge_type', ''))

        evaluation_results['book_knowledge_score'] = book_correct / max(book_questions, 1)
        evaluation_results['paper_knowledge_score'] = paper_correct / max(paper_questions, 1)
        evaluation_results['integration_score'] = integration_correct / max(integration_questions, 1)
        evaluation_results['overall_score'] = sum([book_correct, paper_correct, integration_correct]) / len(test_questions)

        logger.info(f"Evaluation complete - Overall: {evaluation_results['overall_score']:.2f}, "
                   f"Book: {evaluation_results['book_knowledge_score']:.2f}, "
                   f"Paper: {evaluation_results['paper_knowledge_score']:.2f}, "
                   f"Integration: {evaluation_results['integration_score']:.2f}")

        return evaluation_results

    def _evaluate_answer_quality(self, expected: str, generated: str) -> bool:
        """Simple answer quality evaluation (can be enhanced)."""

        expected_lower = expected.lower()
        generated_lower = generated.lower()

        # Check for key terms from expected answer
        expected_words = set(expected_lower.split())
        generated_words = set(generated_lower.split())

        # Simple overlap scoring
        overlap = len(expected_words.intersection(generated_words))
        overlap_ratio = overlap / len(expected_words) if expected_words else 0

        return overlap_ratio > 0.3  # 30% word overlap threshold

    def save_training_results(self, results: Dict, output_path: str = None):
        """Save training results to JSON file."""

        if output_path is None:
            output_path = self.output_dir / "training_results.json"

        # Add metadata
        results['metadata'] = {
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            results['metadata']['gpu_name'] = torch.cuda.get_device_name()
            results['metadata']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Training results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    fine_tuner = IntegratedFineTuner()

    # Example data (normally loaded from processed books and papers)
    book_qa_data = [
        {
            'question': 'What is the structure of benzene?',
            'answer': 'Benzene has a hexagonal ring structure with alternating double bonds.',
            'knowledge_type': 'book_knowledge'
        }
    ]

    paper_qa_data = [
        {
            'question': 'How did Kekulé propose the benzene structure?',
            'answer': 'Kekulé proposed benzene as a hexagonal ring with alternating single and double bonds.',
            'knowledge_type': 'paper_knowledge'
        }
    ]

    integrated_qa_data = [
        {
            'question': 'How does Kekulé\'s original proposal relate to modern understanding of benzene?',
            'answer': 'Kekulé\'s ring structure was foundational, though we now understand benzene has delocalized electrons rather than alternating bonds.',
            'knowledge_type': 'integration_knowledge'
        }
    ]

    print("Enhanced Fine-Tuner initialized")
    print(f"Configuration: {fine_tuner.config}")
    print("Ready for multi-stage training of integrated chemistry knowledge")