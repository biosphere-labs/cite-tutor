"""
Base test classes providing common functionality and mock abstraction.

This module provides base classes that implement the mock abstraction pattern,
allowing tests to run as either unit tests (with mocks) or integration tests
(with real external calls) based on configuration.
"""

import pytest
import os
import unittest
from abc import ABC, abstractmethod
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseTestCase(unittest.TestCase):
    """
    Base test case with common setup and utilities.
    """

    def setUp(self):
        """Set up test environment."""
        self.test_mode = os.getenv("TEST_MODE", "unit")
        self.mock_external_apis = os.getenv("MOCK_EXTERNAL_APIS", "true").lower() == "true"
        self.temp_files = []
        self.patches = []

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        for file_path in self.temp_files:
            if file_path.exists():
                file_path.unlink()

        # Stop all patches
        for patcher in self.patches:
            patcher.stop()

    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> Path:
        """Create a temporary file for testing."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=suffix)
        temp_path = Path(path)
        if content:
            temp_path.write_text(content)
        else:
            os.close(fd)
        self.temp_files.append(temp_path)
        return temp_path

    def patch_method(self, target: str, return_value=None, side_effect=None):
        """Convenience method for patching."""
        patcher = patch(target, return_value=return_value, side_effect=side_effect)
        mock_obj = patcher.start()
        self.patches.append(patcher)
        return mock_obj


class MockableTestCase(BaseTestCase):
    """
    Test case that supports switching between mocked and real implementations.
    """

    def get_implementation(self, module_path: str, class_name: str, mock_factory=None):
        """
        Get either mocked or real implementation based on test configuration.

        Args:
            module_path: Python module path
            class_name: Class name to import/mock
            mock_factory: Function that returns configured mock

        Returns:
            Mock or real class instance
        """
        if self.test_mode == "unit" and self.mock_external_apis:
            if mock_factory:
                return mock_factory()
            else:
                return Mock()
        else:
            # Import and return real implementation
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)


class NetworkTestCase(MockableTestCase):
    """
    Base class for tests that make network requests.
    """

    def setUp(self):
        super().setUp()
        self.mock_session = None
        if self.test_mode == "unit":
            self.setup_network_mocks()

    def setup_network_mocks(self):
        """Set up network-related mocks."""
        self.mock_session = Mock()
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        self.mock_response.text = "<html><body>Test response</body></html>"
        self.mock_response.content = b"Test content"
        self.mock_response.headers = {"Content-Type": "text/html"}
        self.mock_response.json.return_value = {"test": "data"}

        self.mock_session.get.return_value = self.mock_response
        self.mock_session.post.return_value = self.mock_response

        # Patch requests
        self.requests_patcher = patch('requests.Session', return_value=self.mock_session)
        self.requests_patcher.start()
        self.patches.append(self.requests_patcher)

    def configure_response(self, status_code=200, content=None, headers=None, json_data=None):
        """Configure mock response for network calls."""
        if self.mock_response:
            self.mock_response.status_code = status_code
            if content is not None:
                if isinstance(content, str):
                    self.mock_response.text = content
                    self.mock_response.content = content.encode()
                else:
                    self.mock_response.content = content
            if headers:
                self.mock_response.headers = headers
            if json_data:
                self.mock_response.json.return_value = json_data


class FileSystemTestCase(MockableTestCase):
    """
    Base class for tests that interact with the file system.
    """

    def setUp(self):
        super().setUp()
        self.temp_dir = None
        if self.test_mode == "unit":
            self.setup_filesystem_mocks()

    def setup_filesystem_mocks(self):
        """Set up file system mocks."""
        import tempfile
        import shutil
        self.temp_dir = Path(tempfile.mkdtemp())

        # Mock file operations to use temp directory
        self.original_cwd = os.getcwd()

    def tearDown(self):
        super().tearDown()
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)


class MLTestCase(MockableTestCase):
    """
    Base class for machine learning related tests.
    """

    def setUp(self):
        super().setUp()
        if self.test_mode == "unit":
            self.setup_ml_mocks()

    def setup_ml_mocks(self):
        """Set up ML framework mocks."""
        # Mock PyTorch
        self.mock_tensor = Mock()
        self.mock_tensor.shape = [2, 768]
        self.mock_tensor.device = "cpu"
        self.mock_tensor.to.return_value = self.mock_tensor

        torch_patcher = patch('torch.tensor', return_value=self.mock_tensor)
        torch_patcher.start()
        self.patches.append(torch_patcher)

        cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        cuda_patcher.start()
        self.patches.append(cuda_patcher)

        # Mock transformers
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

        self.mock_model.eval.return_value = self.mock_model
        self.mock_model.train.return_value = self.mock_model
        self.mock_model.to.return_value = self.mock_model

        self.mock_tokenizer.return_value = {
            'input_ids': [[101, 2023, 2003, 1037, 3231, 102]],
            'attention_mask': [[1, 1, 1, 1, 1, 1]]
        }
        self.mock_tokenizer.decode.return_value = "Test text"

        model_patcher = patch('transformers.AutoModel.from_pretrained', return_value=self.mock_model)
        model_patcher.start()
        self.patches.append(model_patcher)

        tokenizer_patcher = patch('transformers.AutoTokenizer.from_pretrained', return_value=self.mock_tokenizer)
        tokenizer_patcher.start()
        self.patches.append(tokenizer_patcher)


class PDFTestCase(FileSystemTestCase):
    """
    Base class for PDF processing tests.
    """

    def setUp(self):
        super().setUp()
        if self.test_mode == "unit":
            self.setup_pdf_mocks()

    def setup_pdf_mocks(self):
        """Set up PDF processing mocks."""
        # Mock PyMuPDF
        self.mock_doc = Mock()
        self.mock_page = Mock()

        self.mock_page.get_text.return_value = "Sample PDF text content"
        self.mock_page.get_pixmap.return_value = Mock()
        self.mock_page.number = 0

        self.mock_doc.__len__.return_value = 3
        self.mock_doc.__getitem__.return_value = self.mock_page
        self.mock_doc.__iter__.return_value = iter([self.mock_page] * 3)
        self.mock_doc.page_count = 3

        fitz_patcher = patch('fitz.open', return_value=self.mock_doc)
        fitz_patcher.start()
        self.patches.append(fitz_patcher)

        # Mock Tesseract
        ocr_patcher = patch('pytesseract.image_to_string', return_value="OCR extracted text")
        ocr_patcher.start()
        self.patches.append(ocr_patcher)

        # Mock PIL
        self.mock_image = Mock()
        self.mock_image.size = (800, 600)
        self.mock_image.mode = "RGB"

        pil_patcher = patch('PIL.Image.open', return_value=self.mock_image)
        pil_patcher.start()
        self.patches.append(pil_patcher)


class DatabaseTestCase(MockableTestCase):
    """
    Base class for database-related tests.
    """

    def setUp(self):
        super().setUp()
        if self.test_mode == "unit":
            self.setup_database_mocks()

    def setup_database_mocks(self):
        """Set up database mocks."""
        # Mock database connections and operations
        self.mock_connection = Mock()
        self.mock_cursor = Mock()

        self.mock_cursor.fetchall.return_value = []
        self.mock_cursor.fetchone.return_value = None
        self.mock_cursor.execute.return_value = None

        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection.commit.return_value = None
        self.mock_connection.rollback.return_value = None


# Test Mixins for specific functionality
class AssertionMixin:
    """Mixin providing additional assertion methods."""

    def assertListsAlmostEqual(self, list1: List[float], list2: List[float], places: int = 7):
        """Assert that two lists of floats are almost equal."""
        self.assertEqual(len(list1), len(list2), "Lists have different lengths")
        for i, (a, b) in enumerate(zip(list1, list2)):
            self.assertAlmostEqual(a, b, places=places, msg=f"Lists differ at index {i}")

    def assertDictContainsSubset(self, subset: Dict, dictionary: Dict):
        """Assert that dictionary contains all items in subset."""
        for key, value in subset.items():
            self.assertIn(key, dictionary, f"Key '{key}' not found in dictionary")
            self.assertEqual(dictionary[key], value, f"Value for key '{key}' differs")

    def assertValidResponse(self, response: Dict, required_keys: List[str]):
        """Assert that a response dictionary contains required keys."""
        self.assertIsInstance(response, dict, "Response should be a dictionary")
        for key in required_keys:
            self.assertIn(key, response, f"Required key '{key}' missing from response")

    def assertValidCitation(self, citation: Dict):
        """Assert that a citation dictionary has valid structure."""
        required_keys = ['text', 'authors', 'title', 'year']
        self.assertValidResponse(citation, required_keys)
        self.assertIsInstance(citation['authors'], list, "Authors should be a list")
        self.assertGreater(len(citation['authors']), 0, "Should have at least one author")


class PerformanceMixin:
    """Mixin for performance-related test utilities."""

    def assertExecutionTime(self, func, max_time: float, *args, **kwargs):
        """Assert that function executes within specified time."""
        import time
        start = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start
        self.assertLessEqual(
            execution_time, max_time,
            f"Function took {execution_time:.3f}s, expected < {max_time}s"
        )
        return result

    def assertMemoryUsage(self, func, max_memory_mb: float, *args, **kwargs):
        """Assert that function uses less than specified memory."""
        import tracemalloc
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        self.assertLessEqual(
            peak_mb, max_memory_mb,
            f"Function used {peak_mb:.2f}MB, expected < {max_memory_mb}MB"
        )
        return result


# Combined base classes for common use cases
class StandardTestCase(BaseTestCase, AssertionMixin, PerformanceMixin):
    """Standard test case with common mixins."""
    pass


class NetworkStandardTestCase(NetworkTestCase, AssertionMixin, PerformanceMixin):
    """Network test case with common mixins."""
    pass


class MLStandardTestCase(MLTestCase, AssertionMixin, PerformanceMixin):
    """ML test case with common mixins."""
    pass


class PDFStandardTestCase(PDFTestCase, AssertionMixin, PerformanceMixin):
    """PDF test case with common mixins."""
    pass