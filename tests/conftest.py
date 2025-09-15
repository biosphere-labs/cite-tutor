"""
Pytest configuration and shared fixtures.

This module provides:
- Test configuration and setup
- Shared fixtures for all test modules
- Mock abstraction system for unit/integration test switching
- Common test utilities and helpers
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Test configuration
TEST_MODE = os.getenv("TEST_MODE", "unit")
MOCK_EXTERNAL_APIS = os.getenv("MOCK_EXTERNAL_APIS", "true").lower() == "true"

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "test_mode": TEST_MODE,
        "mock_external_apis": MOCK_EXTERNAL_APIS,
        "testing": True,
        "disable_gpu": True,
        "log_level": "WARNING"
    }

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def test_data_dir():
    """Test data directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Path to sample PDF for testing."""
    pdf_path = test_data_dir / "sample_academic_paper.pdf"
    if not pdf_path.exists():
        # Create a minimal PDF for testing
        create_sample_pdf(pdf_path)
    return pdf_path

@pytest.fixture
def sample_text_content():
    """Sample academic text content for testing."""
    return """
    # Introduction to Quantum Mechanics

    Quantum mechanics is a fundamental theory in physics that describes the behavior
    of matter and energy at atomic and subatomic scales. The theory was developed in
    the early 20th century and has been extensively tested.

    ## Key Principles

    1. Wave-particle duality: Particles exhibit both wave and particle properties
    2. Uncertainty principle: Position and momentum cannot be simultaneously known
    3. Superposition: Quantum systems can exist in multiple states simultaneously

    The Schrödinger equation describes the evolution of quantum systems:

    iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩

    ## References

    1. Heisenberg, W. (1927). "Über den anschaulichen Inhalt der quantentheoretischen
       Kinematik und Mechanik". Zeitschrift für Physik. 43 (3–4): 172–198.
    2. Schrödinger, E. (1926). "An Undulatory Theory of the Mechanics of Atoms and
       Molecules". Physical Review. 28 (6): 1049–1070.
    """

# Mock Abstraction System
class MockManager:
    """
    Central mock manager that can switch between mocked and real implementations
    based on test configuration.
    """

    def __init__(self, test_mode: str = "unit"):
        self.test_mode = test_mode
        self.mocks = {}

    def get_mock_or_real(self, module_path: str, class_name: str, mock_impl=None):
        """
        Returns either a mock implementation or the real class based on test mode.

        Args:
            module_path: Python module path (e.g., 'requests')
            class_name: Class name to mock (e.g., 'Session')
            mock_impl: Custom mock implementation

        Returns:
            Mock object or real class
        """
        if self.test_mode == "unit" and MOCK_EXTERNAL_APIS:
            if mock_impl:
                return mock_impl
            else:
                return Mock()
        else:
            # Return real implementation for integration tests
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

@pytest.fixture
def mock_manager(test_config):
    """Mock manager instance for controlling mocks vs real implementations."""
    return MockManager(test_config["test_mode"])

# External API Mocks
@pytest.fixture
def mock_requests(mock_manager):
    """Mock or real requests session."""
    if mock_manager.test_mode == "unit":
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Test response</body></html>"
        mock_response.content = b"Test PDF content"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.json.return_value = {"test": "data"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response

        with patch('requests.Session', return_value=mock_session):
            with patch('requests.get', return_value=mock_response):
                with patch('requests.post', return_value=mock_response):
                    yield mock_session
    else:
        import requests
        yield requests.Session()

@pytest.fixture
def mock_transformers(mock_manager):
    """Mock or real transformers components."""
    if mock_manager.test_mode == "unit":
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_trainer = Mock()

        # Mock model behavior
        mock_model.return_value.logits = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.train.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {
            'input_ids': [[101, 2023, 2003, 1037, 3231, 102]],
            'attention_mask': [[1, 1, 1, 1, 1, 1]]
        }
        mock_tokenizer.decode.return_value = "This is a test"
        mock_tokenizer.encode.return_value = [101, 2023, 2003, 1037, 3231, 102]

        # Mock trainer behavior
        mock_trainer.train.return_value = None
        mock_trainer.evaluate.return_value = {"eval_loss": 0.5, "eval_accuracy": 0.85}

        with patch('transformers.AutoModel.from_pretrained', return_value=mock_model):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                with patch('transformers.Trainer', return_value=mock_trainer):
                    yield {
                        'model': mock_model,
                        'tokenizer': mock_tokenizer,
                        'trainer': mock_trainer
                    }
    else:
        import transformers
        yield transformers

@pytest.fixture
def mock_torch(mock_manager):
    """Mock or real PyTorch components."""
    if mock_manager.test_mode == "unit":
        mock_tensor = Mock()
        mock_tensor.shape = [2, 768]
        mock_tensor.device = "cpu"
        mock_tensor.dtype = "float32"

        with patch('torch.tensor', return_value=mock_tensor):
            with patch('torch.save') as mock_save:
                with patch('torch.load', return_value={'model': 'state_dict'}):
                    with patch('torch.cuda.is_available', return_value=False):
                        yield {
                            'tensor': mock_tensor,
                            'save': mock_save,
                            'cuda_available': False
                        }
    else:
        import torch
        yield torch

@pytest.fixture
def mock_pdf_tools(mock_manager):
    """Mock or real PDF processing tools."""
    if mock_manager.test_mode == "unit":
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample PDF text content"
        mock_page.get_pixmap.return_value = Mock()
        mock_doc.__len__.return_value = 3
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__iter__.return_value = iter([mock_page, mock_page, mock_page])

        # Mock Tesseract
        mock_ocr_result = "OCR extracted text from image"

        with patch('fitz.open', return_value=mock_doc):
            with patch('pytesseract.image_to_string', return_value=mock_ocr_result):
                yield {
                    'doc': mock_doc,
                    'page': mock_page,
                    'ocr_result': mock_ocr_result
                }
    else:
        import fitz
        import pytesseract
        yield {
            'fitz': fitz,
            'pytesseract': pytesseract
        }

# Test Data Creation Utilities
def create_sample_pdf(pdf_path: Path):
    """Create a minimal PDF file for testing."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        text = """
        Sample Academic Paper

        This is a test document for the sci-tutor system.
        It contains some academic content for testing purposes.

        Key concepts:
        - Machine learning
        - Natural language processing
        - Educational technology

        References:
        1. Smith, J. (2023). "AI in Education". Journal of Educational Technology.
        2. Brown, A. (2022). "Natural Language Processing for Learning". ACM Computing Surveys.
        """
        page.insert_text((72, 72), text)
        doc.save(str(pdf_path))
        doc.close()
    except ImportError:
        # Create a dummy file if fitz is not available
        pdf_path.write_bytes(b"%PDF-1.4\n%Test PDF for testing\n%%EOF")

@pytest.fixture
def sample_citations():
    """Sample citations for testing."""
    return [
        {
            "text": "Smith, J. (2023). Artificial Intelligence in Education: A Comprehensive Review. Journal of Educational Technology, 45(2), 123-145.",
            "authors": ["Smith, J."],
            "title": "Artificial Intelligence in Education: A Comprehensive Review",
            "journal": "Journal of Educational Technology",
            "year": "2023",
            "volume": "45",
            "issue": "2",
            "pages": "123-145"
        },
        {
            "text": "Brown, A., & Johnson, K. (2022). Natural Language Processing for Personalized Learning. In Proceedings of the Conference on Educational Data Mining (pp. 67-82).",
            "authors": ["Brown, A.", "Johnson, K."],
            "title": "Natural Language Processing for Personalized Learning",
            "journal": "Proceedings of the Conference on Educational Data Mining",
            "year": "2022",
            "pages": "67-82"
        }
    ]

@pytest.fixture
def sample_training_data():
    """Sample training data for ML models."""
    return {
        "texts": [
            "What is the definition of photosynthesis?",
            "Explain Newton's second law of motion.",
            "How does the water cycle work?",
            "What are the main components of a cell?"
        ],
        "labels": [0, 1, 0, 2],  # Different subject categories
        "answers": [
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Newton's second law states that F = ma, where force equals mass times acceleration.",
            "The water cycle involves evaporation, condensation, precipitation, and collection.",
            "The main components of a cell include the nucleus, cytoplasm, and cell membrane."
        ]
    }

# Test Utilities
@pytest.fixture
def assert_called_with_retry():
    """Utility for testing retried operations."""
    def _assert_called_with_retry(mock_obj, expected_calls, max_retries=3):
        """Assert that a mock was called with expected arguments, accounting for retries."""
        actual_calls = mock_obj.call_args_list
        assert len(actual_calls) <= max_retries * len(expected_calls)

        # Check that expected calls appear in actual calls
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    return _assert_called_with_retry

@pytest.fixture
def wait_for_condition():
    """Utility for waiting for async conditions."""
    import time

    def _wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

    return _wait_for_condition

# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real calls")
    config.addinivalue_line("markers", "slow: Tests that take more than 5 seconds")
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
    config.addinivalue_line("markers", "network: Tests that require network access")
    config.addinivalue_line("markers", "expensive: Tests that use external APIs")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and environment."""
    if TEST_MODE == "unit":
        skip_integration = pytest.mark.skip(reason="Running in unit test mode")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    elif TEST_MODE == "integration":
        skip_unit = pytest.mark.skip(reason="Running in integration test mode")
        for item in items:
            if "unit" in item.keywords and "integration" not in item.keywords:
                item.add_marker(skip_unit)