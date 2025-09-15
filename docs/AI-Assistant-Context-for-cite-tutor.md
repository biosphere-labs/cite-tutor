# AI Assistant Context for cite-tutor

## Project Overview

**cite-tutor** is an advanced AI-powered academic research assistant designed to process PDF books and papers, extract and validate citations, retrieve foundational papers via Google Scholar + Sci-Hub, and create a fine-tuned AI tutoring system with real-time citation lookup capabilities. The project is specifically optimized for 4GB VRAM GPUs and supports multiple academic domains.

### Core Purpose
Unlike generic AI assistants that may hallucinate citations, cite-tutor grounds its responses in verified academic sources. It traces knowledge back to original papers, validates citations in real-time, and builds domain expertise from foundational literature. This makes it invaluable for researchers, students, and academics who need accurate, citable information with provenance.

### Key Innovation
The system can be deployed as an **MCP (Model Context Protocol) plugin**, allowing integration with primary AI systems like OpenAI or Claude to enhance them with verified academic knowledge and citation capabilities.

## Python Learning Opportunities

This project offers excellent opportunities to learn Python across multiple domains:

### 1. **Machine Learning & AI** (`src/enhanced_fine_tuner.py`, `src/paper_processor.py`)
- **Transformers Library**: Uses HuggingFace transformers for T5 and GPT models
- **PyTorch**: Deep learning framework for model training and inference
- **PEFT & LoRA**: Parameter-efficient fine-tuning techniques
- **Memory Management**: GPU memory optimization for 4GB constraints
- **Quantization**: 4-bit quantization for efficient model deployment

**Learning Topics**:
- Model fine-tuning with limited resources
- Memory optimization techniques
- Transformer architecture implementation
- Custom training callbacks and monitoring

### 2. **Document Processing** (`src/pdf_processor.py`, `src/structure_detector.py`)
- **PyMuPDF**: Advanced PDF text extraction and manipulation
- **Tesseract/Pytesseract**: OCR for scanned documents
- **PIL/Pillow**: Image processing and enhancement
- **Regular Expressions**: Complex pattern matching for academic content

**Learning Topics**:
- PDF text extraction from various formats
- OCR preprocessing and optimization
- Image processing for document enhancement
- Academic document structure detection

### 3. **Web Scraping & API Integration** (`src/scholar_scraper.py`, `src/scihub_client.py`)
- **Requests**: HTTP client for web scraping
- **BeautifulSoup**: HTML parsing and data extraction
- **Selenium**: Browser automation (if needed)
- **Rate limiting and ethical scraping**: Responsible API usage

**Learning Topics**:
- Web scraping best practices
- API rate limiting and retries
- HTML parsing and data extraction
- Handling dynamic content

### 4. **Data Structures & Configuration** (`src/domain_config.py`, `config/domains.yaml`)
- **YAML**: Configuration file parsing and management
- **Dataclasses**: Modern Python data structure definitions
- **Type Hints**: Comprehensive type annotation usage
- **Factory Pattern**: Domain-specific configuration management

**Learning Topics**:
- Configuration-driven development
- Type-safe Python programming
- Dataclass best practices
- Domain-driven design patterns

### 5. **Testing & Quality Assurance** (`tests/`)
- **Pytest**: Modern testing framework
- **Fixtures**: Test data and setup management
- **Mocking**: Isolating components for unit testing
- **Coverage**: Test coverage analysis

**Learning Topics**:
- Test-driven development
- Unit testing best practices
- Mocking external dependencies
- Testing ML pipelines

## Architecture Deep Dive

### Core Components

#### 1. **PDF Processing Pipeline**
```python
# Main entry point: src/pdf_processor.py
class PDFExtractor:
    def extract_text(self, pdf_path: str) -> Dict[str, Any]
    def _extract_with_ocr(self, page) -> str
    def _enhance_image_quality(self, image) -> Image
```

**Key Learning Points**:
- Handles both modern PDFs and scanned documents
- Implements caching for expensive OCR operations
- Uses academic-specific validation patterns
- Demonstrates error handling and logging best practices

#### 2. **Citation Extraction System**
```python
# Domain-aware citation extraction: src/citation_extractor.py
@dataclass
class ChemistryCitation:
    citation_text: str
    citation_type: str
    context: str
    chemistry_context: str
    importance_score: float
    # ... additional fields
```

**Key Learning Points**:
- Dataclass usage for structured data
- Domain-specific pattern matching
- Complex regex patterns for academic citation formats
- Scoring and ranking algorithms

#### 3. **Multi-Domain Configuration System**
```python
# Centralized configuration: src/domain_config.py
class DomainConfiguration:
    def __init__(self, config_path: str = None, default_domain: str = None)
    def get_domain_config(self, domain: str) -> Dict[str, Any]
    def get_journals(self, domain: str) -> Dict[str, str]
    def get_foundational_authors(self, domain: str) -> List[str]
```

**Key Learning Points**:
- Configuration-driven architecture
- YAML file parsing and validation
- Factory pattern implementation
- Error handling for missing configurations

#### 4. **AI Training Pipeline**
```python
# Memory-optimized training: src/enhanced_fine_tuner.py
class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs)

class EnhancedFineTuner:
    def setup_model_for_training(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]
    def fine_tune_integrated_knowledge(self, book_data: List[Dict], paper_data: List[Dict])
```

**Key Learning Points**:
- Custom training callbacks
- Memory management for GPU constraints
- Parameter-efficient fine-tuning (LoRA)
- Multi-stage training workflows

### 5. **GPU Memory Optimization**
```python
# Memory validation: src/utils/gpu_validator.py
def check_gpu_memory() -> Dict[str, Any]
def estimate_model_memory(model_name: str) -> float
def validate_memory_config(config: Dict) -> bool
```

**Key Learning Points**:
- GPU memory monitoring
- Model memory estimation
- Dynamic resource allocation
- Performance optimization strategies

## Configuration Files

### 1. **Model Configuration** (`config/models.yaml`)
Defines 4GB VRAM optimized models:
- Structure analysis: FLAN-T5-small (60MB)
- Fine-tuning: DistilGPT2 (82MB) with 4-bit quantization
- Embeddings: all-MiniLM-L6-v2 (22MB)
- Memory limits and safety buffers

### 2. **Domain Configuration** (`config/domains.yaml`)
Comprehensive domain-specific configurations:
- 5 supported domains: chemistry, physics, mathematics, biology, engineering
- Journal abbreviations and full names
- Domain-specific keywords and patterns
- Foundational authors and historical context
- Content classification patterns
- Sample training data

## Project Structure & Learning Path

```
cite-tutor/
├── src/                          # Core Python modules
│   ├── pdf_processor.py          # 🎯 START HERE: Document processing
│   ├── citation_extractor.py     # 🎯 NEXT: Pattern matching & regex
│   ├── domain_config.py          # 🎯 THEN: Configuration management
│   ├── paper_processor.py        # 🎯 Advanced: ML data processing
│   ├── enhanced_fine_tuner.py    # 🎯 Expert: ML model training
│   ├── scholar_scraper.py        # Web scraping & APIs
│   ├── scihub_client.py          # HTTP client implementation
│   ├── structure_detector.py     # AI-powered document analysis
│   ├── training_manager.py       # Training pipeline orchestration
│   └── utils/                    # Utility functions
│       ├── gpu_validator.py      # Hardware validation
│       └── __init__.py          # Package initialization
├── config/                       # Configuration files
│   ├── models.yaml              # Model configurations
│   └── domains.yaml             # Domain-specific settings
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest configuration
│   ├── test_*.py               # Unit tests for each module
└── docs/                        # Documentation
    └── [various .md files]      # Learning materials
```

## Suggested Learning Progression

### **Beginner Level** (Start with these files)
1. **`src/utils/gpu_validator.py`** - Simple utility functions, basic Python concepts
2. **`config/domains.yaml`** - YAML structure, configuration patterns
3. **`src/domain_config.py`** - Classes, error handling, file I/O

### **Intermediate Level** (Build on basics)
4. **`src/pdf_processor.py`** - External libraries, complex data processing
5. **`src/citation_extractor.py`** - Regular expressions, dataclasses, algorithms
6. **`tests/test_*.py`** - Unit testing, pytest, mocking

### **Advanced Level** (ML and AI concepts)
7. **`src/paper_processor.py`** - Machine learning pipelines, data preprocessing
8. **`src/enhanced_fine_tuner.py`** - Deep learning, model training, GPU optimization
9. **`src/scholar_scraper.py`** - Web scraping, API integration, async programming

## Key Python Concepts Demonstrated

### **Modern Python Features**
- **Type Hints**: Comprehensive type annotations throughout
- **Dataclasses**: Clean data structure definitions
- **Pathlib**: Modern file path handling
- **F-strings**: Modern string formatting
- **Context Managers**: Resource management (`with` statements)
- **Generators**: Memory-efficient data processing

### **Object-Oriented Programming**
- **Classes and Inheritance**: Well-structured class hierarchies
- **Encapsulation**: Private methods and data hiding
- **Composition**: Building complex objects from simpler ones
- **Factory Pattern**: Domain configuration management

### **Error Handling**
- **Custom Exceptions**: Domain-specific error types
- **Try-Catch Blocks**: Robust error handling
- **Logging**: Comprehensive logging throughout the application
- **Validation**: Input validation and sanitation

### **Concurrency & Performance**
- **Memory Management**: GPU memory optimization
- **Caching**: Results caching for expensive operations
- **Batch Processing**: Efficient data processing strategies
- **Resource Pooling**: Managing limited GPU resources

## Development Environment Setup

### **Dependencies** (from `setup.py`)
```python
install_requires=[
    "transformers==4.30.2",        # 🤖 Hugging Face transformers
    "torch>=2.0.1",               # 🔥 PyTorch deep learning
    "datasets==2.14.0",           # 📊 Dataset management
    "PyMuPDF==1.23.0",            # 📄 PDF processing
    "pytesseract==0.3.10",        # 👁️ OCR capabilities
    "beautifulsoup4==4.12.2",     # 🌐 Web scraping
    "chromadb==0.4.0",            # 🗄️ Vector database
    "sentence-transformers==2.2.2", # 🔤 Text embeddings
    "peft==0.4.0",                # ⚡ Parameter-efficient fine-tuning
    # ... and more
]
```

### **Development Tools**
```python
extras_require={
    "dev": [
        "pytest>=7.0.0",          # 🧪 Testing framework
        "black>=23.0.0",          # 🖤 Code formatting
        "flake8>=6.0.0",          # 🔍 Code linting
        "mypy>=1.0.0",            # 📝 Type checking
    ]
}
```

## Hands-on Learning Exercises

### **Exercise 1: Configuration Management**
**Goal**: Understand YAML parsing and configuration patterns
**Task**: Extend `domains.yaml` to add a new academic domain (e.g., "psychology")
**Files to modify**:
- `config/domains.yaml`
- `src/domain_config.py`

**Learning outcomes**:
- YAML file structure
- Dictionary manipulation
- Error handling for missing keys

### **Exercise 2: Data Processing Pipeline**
**Goal**: Learn document processing and text extraction
**Task**: Add support for a new file format (e.g., DOCX) to the PDF processor
**Files to modify**:
- `src/pdf_processor.py`
- Add tests in `tests/test_pdf_processor.py`

**Learning outcomes**:
- External library integration
- Polymorphism and abstraction
- Unit testing strategies

### **Exercise 3: Pattern Matching & Regex**
**Goal**: Master regular expressions and text pattern matching
**Task**: Improve citation extraction patterns for a specific journal format
**Files to modify**:
- `src/citation_extractor.py`
- Update corresponding domain configuration

**Learning outcomes**:
- Advanced regex patterns
- Text processing algorithms
- Domain-specific knowledge encoding

### **Exercise 4: Machine Learning Pipeline**
**Goal**: Understand ML model training and optimization
**Task**: Add a new model checkpoint saving strategy
**Files to modify**:
- `src/enhanced_fine_tuner.py`
- `config/models.yaml`

**Learning outcomes**:
- ML model lifecycle management
- GPU memory optimization
- Callback pattern implementation

## Common Python Patterns in the Codebase

### **1. Configuration Factory Pattern**
```python
def get_domain_config(domain: str = None) -> DomainConfiguration:
    """Factory function for domain configuration."""
    return DomainConfiguration(default_domain=domain or "chemistry")
```

### **2. Context Manager Pattern**
```python
class MemoryManager:
    def __enter__(self):
        self.initial_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
```

### **3. Callback Pattern**
```python
class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Monitor memory usage during training
        pass
```

### **4. Builder Pattern**
```python
class FineTunerBuilder:
    def with_model(self, model_name: str):
        self.model_name = model_name
        return self

    def with_memory_limit(self, limit: int):
        self.memory_limit = limit
        return self

    def build(self) -> EnhancedFineTuner:
        return EnhancedFineTuner(self.model_name, self.memory_limit)
```

## Performance Considerations

### **Memory Optimization Strategies**
1. **4GB VRAM Constraint**: All models selected for minimal memory footprint
2. **Quantization**: 4-bit quantization reduces model size by 75%
3. **LoRA**: Parameter-efficient fine-tuning reduces trainable parameters by 99%
4. **Batch Size**: Optimized batch sizes (typically 1) for memory constraints
5. **Gradient Accumulation**: Simulates larger batches without memory overhead

### **Computational Efficiency**
1. **Caching**: Expensive OCR operations are cached
2. **Lazy Loading**: Models loaded only when needed
3. **Memory Monitoring**: Real-time GPU memory tracking
4. **Early Stopping**: Training stops when memory limits approached

## Testing Strategy

### **Test Structure**
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Fixtures**: Reusable test data and configurations
- **Mocking**: External service mocking (Sci-Hub, Google Scholar)

### **Test Files**
- `tests/test_pdf_processor.py` - Document processing tests
- `tests/test_citation_extractor.py` - Citation extraction tests
- `tests/test_paper_processor.py` - ML pipeline tests
- `tests/conftest.py` - Shared test configuration

## Documentation & Learning Resources

### **Internal Documentation**
- Comprehensive README with setup instructions
- Detailed module docstrings
- Type hints for all functions and classes
- Inline comments explaining complex algorithms

### **Training Materials** (`docs/` folder)
- Training fundamentals
- Model architecture explanations
- Evaluation and optimization guides
- Production deployment strategies
- Assessment materials and exams

### **Configuration Examples**
- Complete YAML configurations for all domains
- Model configuration examples
- Memory optimization settings

## Getting Started as a Python Learner

### **1. Start with the Basics**
```bash
# Set up the environment
conda env create -f environment.yml
conda activate cite-tutor
pip install -e .
```

### **2. Explore the Configuration System**
- Read through `config/domains.yaml` to understand the data structure
- Examine `src/domain_config.py` to see how configuration is loaded
- Try modifying a domain configuration and see how it affects the system

### **3. Run Simple Components**
```bash
# Test GPU validation
python -c "from src.utils.gpu_validator import check_gpu_memory; check_gpu_memory()"

# Test PDF processing on a sample file
python src/pdf_processor.py --input "sample.pdf" --output "output/"
```

### **4. Explore the Test Suite**
```bash
# Run tests to understand expected behavior
pytest tests/test_pdf_processor.py -v
pytest tests/test_citation_extractor.py -v
```

### **5. Gradually Tackle Complex Components**
- Start with utility functions
- Move to data processing classes
- Eventually explore ML training pipelines

## Integration Opportunities

### **MCP Plugin Development**
The project can be extended as an MCP (Model Context Protocol) plugin:
- Real-time citation lookup for AI assistants
- Academic knowledge grounding for generic AI models
- Verified source attribution for AI responses

### **API Development**
Consider building REST APIs around core functionality:
- PDF processing endpoints
- Citation extraction services
- Domain-specific knowledge queries
- Model fine-tuning interfaces

### **Academic Tool Integration**
- Zotero plugin for citation management
- LaTeX integration for academic writing
- Reference management system integration
- Academic search engine enhancement

## Summary

The cite-tutor project represents an excellent opportunity to learn Python across multiple domains: machine learning, document processing, web scraping, configuration management, and testing. The codebase demonstrates modern Python practices, comprehensive type hints, robust error handling, and performance optimization techniques specifically for resource-constrained environments.

The project's structure allows for gradual learning progression, from simple utility functions to complex ML pipelines, making it ideal for both beginner and advanced Python learners. The comprehensive documentation, test suite, and real-world application make it a valuable learning resource for anyone interested in Python development, machine learning, or academic research tools.