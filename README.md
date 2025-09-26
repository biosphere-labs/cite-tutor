# Cite-Tutor

> **Status: Experimental / Work in Progress**
>
> This project was built as a learning exercise exploring PDF processing, citation extraction, web scraping, ML fine-tuning, and MCP integration. It is not actively maintained but demonstrates the architectural approach to solving AI citation hallucination.

Cite-Tutor is an AI-powered academic research assistant that processes PDF books and papers, extracts and validates citations, retrieves foundational papers via Google Scholar + Sci-Hub, and creates a fine-tuned AI tutoring system with real-time citation lookup capabilities. Optimized for 4GB VRAM GPUs and supports any academic domain.

## Why Cite-Tutor?

Unlike generic AI assistants that hallucinate citations or provide outdated information, **Cite-Tutor grounds its responses in verified academic sources**. It doesn't just generate answers -- it traces knowledge back to original papers, validates citations in real-time, and builds domain expertise from foundational literature. This makes it invaluable for researchers, students, and academics who need **accurate, citable information with provenance**. The system trains on domain-specific books plus retrieved foundational papers, creating AI tutors that understand the historical context and evolution of ideas. Additionally, Cite-Tutor can be deployed as an **MCP (Model Context Protocol) plugin**, allowing integration with primary AI systems like OpenAI or Claude to enhance them with verified academic knowledge and citation capabilities.

## Architecture Overview

![Cite-Tutor Architecture](./docs/assets/Cite-Tutor%20Architecture.png)

**Architecture Components:**
- **Local Development**: CLI tools, configuration files, training data preparation
- **AWS Orchestration**: Training pipeline management, job scheduling, cost monitoring
- **Spot Fleet Workers**: GPU-enabled instances for cost-effective model training
- **Storage Layer**: S3 buckets for data, models, and checkpoints
- **Domain System**: Multi-domain support (Chemistry, Physics, Math, Biology, Engineering)
- **Training Pipeline**: 3-stage fine-tuning with specialized models

**View Options:**
- 📊 [Interactive Diagram](https://www.plantuml.com/plantuml/uml/~1UDfTK0kqLSeXFhIKmbAeEllFpKSY60)
- 📁 [Source PlantUML File](architecture.puml)
- 📖 [Detailed Architecture Documentation](docs/aws-spot-fleet-training.md)

## Key Features

- PDF processing with OCR for scanned books and papers
- AI-powered document structure detection
- Citation extraction and paper retrieval
- Fine-tuning on domain-specific books + retrieved papers as core knowledge
- RAG system for real-time citation lookup and tutoring
- 4GB VRAM optimization throughout
- Multi-domain support (science, mathematics, engineering, etc.)

## Hardware Requirements

- **GPU**: 4GB VRAM minimum (RTX 1650, GTX 1050 Ti, or better)
- **RAM**: 8GB+ system memory recommended
- **Storage**: 10GB+ free space for models and data

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/biosphere-labs/cite-tutor.git
cd cite-tutor
```

### 2. Install with UV (Recommended)

```bash
# Install UV if you haven't already
pip install uv

# Install dependencies and package
uv sync --dev
```

### 3. Alternative: Conda Environment

```bash
conda env create -f environment.yml
conda activate cite-tutor
pip install -e .
```

### 4. Verify GPU Memory

```bash
python -c "from cite_tutor.utils.gpu_validator import check_gpu_memory; check_gpu_memory()"
```

## Project Structure

```
cite-tutor/
├── cite_tutor/                   # Main package
│   ├── __init__.py              # Package initialization
│   ├── pdf_processor.py         # PDF extraction with OCR
│   ├── structure_detector.py    # AI document structure (tiny models)
│   ├── citation_extractor.py   # Extract citations from academic books
│   ├── scholar_scraper.py      # Google Scholar DOI lookup
│   ├── scihub_client.py        # Sci-Hub paper retrieval
│   ├── paper_processor.py      # Process retrieved papers as core knowledge
│   ├── enhanced_fine_tuner.py  # Multi-stage fine-tuning (4GB optimized)
│   ├── training_manager.py     # Training pipeline orchestration
│   └── utils/                   # Utility modules
│       ├── __init__.py         # Utils package init
│       └── gpu_validator.py    # GPU memory validation
├── config/
│   ├── models.yaml             # 4GB-optimized model configs
│   └── domains.yaml           # Domain-specific configurations
├── tests/                      # Test suite
├── data/
│   ├── books/                  # Input PDF books
│   ├── structured/             # Processed book structures
│   ├── papers/                 # Retrieved papers cache
│   └── training/               # Generated training data
├── outputs/
│   └── models/                 # Fine-tuned models
├── pyproject.toml              # UV/Python packaging configuration
└── README.md                   # This file
```

**Note:** A `src/` directory exists containing only stale `__pycache__` files from an earlier project layout. The canonical source code lives in `cite_tutor/`.

## Quick Start

1. **Place PDF books** in `data/books/`
2. **Run the full pipeline**:
   ```bash
   cite-tutor --input data/books/ --output outputs/
   # Or using Python directly:
   python -m cite_tutor --input data/books/ --output outputs/
   ```

## Memory Optimization

This project is specifically optimized for 4GB VRAM GPUs:

- **Models**: Uses smallest possible models (distilGPT2, FLAN-T5-small, MiniLM)
- **Quantization**: 4-bit quantization for all fine-tuning
- **LoRA**: Parameter-efficient fine-tuning
- **Batch Size**: Optimized batch sizes (typically 1)
- **Memory Management**: Automatic GPU memory monitoring

## Configuration

Edit `config/models.yaml` to adjust model settings:

- Model selection (always prioritizing smallest models)
- Memory limits and safety buffers
- Fine-tuning parameters (LoRA ranks, batch sizes)
- Quantization settings

## Usage Examples

### Process a Single PDF Book
```bash
process-pdf --input "data/books/academic_textbook.pdf" --output "data/structured/"
# Or using UV:
uv run process-pdf --input "data/books/academic_textbook.pdf" --output "data/structured/"
```

### Extract Citations
```bash
extract-citations --input "data/structured/" --output "data/citations/"
# Or using UV:
uv run extract-citations --input "data/structured/" --output "data/citations/"
```

### Fine-tune the AI Model
```bash
fine-tune --config config/models.yaml --data data/training/
# Or using UV:
uv run fine-tune --config config/models.yaml --data data/training/
```

### Start RAG System
```bash
start-rag --serve --port 8000
# Or using UV:
uv run start-rag --serve --port 8000
```

## Troubleshooting

### GPU Memory Issues
- Reduce batch size in `config/models.yaml`
- Enable 4-bit quantization
- Monitor memory with built-in utilities

### OCR Problems
- Ensure Tesseract is installed: `conda install tesseract`
- Check PDF quality and preprocessing

### Citation Extraction
- Verify PDF structure detection accuracy
- Adjust extraction patterns for specific academic formats

## Documentation

### Core Training & Development Guides
- [Training Fundamentals](docs/01-training-fundamentals.md)
- [Model Architectures](docs/02-model-architectures.md)
- [Training Pipeline](docs/03-training-pipeline.md)
- [Evaluation & Optimization](docs/04-evaluation-optimization.md)
- [Production Deployment](docs/05-production-deployment.md)
- [Debugging & Monitoring](docs/06-debugging-monitoring.md)

### Advanced Topics
- [Reinforcement Learning Optimization](docs/reinforcement-learning-optimization.md)
- [Chemistry to Psi Tutor Changes](docs/chemistry-to-psi-tutor-changes.md)

### Assessment Materials
- [Training Fundamentals Exam](docs/assessments/01-training-fundamentals-exam.md)
- [Model Architectures Exam](docs/assessments/02-model-architectures-exam.md)
- [Training Pipeline Exam](docs/assessments/03-training-pipeline-exam.md)
- [Evaluation & Optimization Exam](docs/assessments/04-evaluation-optimization-exam.md)
- [Production Deployment Exam](docs/assessments/05-production-deployment-exam.md)
- [Debugging & Monitoring Exam](docs/assessments/06-debugging-monitoring-exam.md)
- [Hands-on Challenges](docs/assessments/hands-on-challenges.md)

## License

MIT License - see LICENSE file for details.
