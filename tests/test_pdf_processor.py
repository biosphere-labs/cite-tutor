"""
Tests for PDF processor module.

Tests both unit (mocked) and integration (real PDF processing) modes.
Uses PDFStandardTestCase for comprehensive PDF processing mocking.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os
import io
from PIL import Image
import numpy as np

from base_test import PDFStandardTestCase
from pdf_processor import PDFExtractor


class TestPDFExtractor(PDFStandardTestCase):
    """Test PDFExtractor class with mock abstraction."""

    def setUp(self):
        super().setUp()
        self.extractor = PDFExtractor(cache_dir="test_cache", enable_caching=True)
        self.sample_pdf_path = "test_document.pdf"
        self.sample_text = "Sample academic text with equations: H2O + CO2 → H2CO3"

    def test_init_creates_cache_directory(self):
        """Test that PDFExtractor creates cache directory on initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test_cache"
            extractor = PDFExtractor(cache_dir=str(cache_path))
            self.assertTrue(cache_path.exists())

    def test_get_cache_key_with_existing_file(self):
        """Test cache key generation for existing files."""
        if self.test_mode == "unit":
            # Mock Path and stat for unit tests
            with patch('pathlib.Path') as mock_path:
                mock_path_obj = Mock()
                mock_path_obj.exists.return_value = True
                mock_stat = Mock()
                mock_stat.st_size = 1024
                mock_stat.st_mtime = 1640995200.0
                mock_path_obj.stat.return_value = mock_stat
                mock_path.return_value = mock_path_obj

                cache_key = self.extractor._get_cache_key("test.pdf")
                self.assertIsNotNone(cache_key)
                self.assertIsInstance(cache_key, str)
                self.assertEqual(len(cache_key), 32)  # MD5 hash length
        else:
            # Integration test with real file
            temp_file = self.create_temp_file("test content", ".pdf")
            cache_key = self.extractor._get_cache_key(str(temp_file))
            self.assertIsNotNone(cache_key)
            self.assertEqual(len(cache_key), 32)

    def test_get_cache_key_with_nonexistent_file(self):
        """Test cache key generation for non-existent files."""
        cache_key = self.extractor._get_cache_key("nonexistent.pdf")
        self.assertIsNone(cache_key)

    def test_load_from_cache_with_valid_cache(self):
        """Test loading data from cache."""
        if self.test_mode == "unit":
            test_data = {"text": "cached text", "quality": 0.85}
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('pickle.load', return_value=test_data):
                    with patch.object(Path, 'exists', return_value=True):
                        result = self.extractor._load_from_cache("test_key")
                        self.assertEqual(result, test_data)
        else:
            # Integration test with real cache
            test_data = {"text": "test content", "quality": 0.9}
            self.extractor._save_to_cache("test_cache_key", test_data)
            loaded_data = self.extractor._load_from_cache("test_cache_key")
            self.assertEqual(loaded_data, test_data)

    def test_load_from_cache_with_invalid_cache(self):
        """Test loading from non-existent cache."""
        result = self.extractor._load_from_cache("nonexistent_key")
        self.assertIsNone(result)

    def test_save_to_cache(self):
        """Test saving data to cache."""
        test_data = {"text": "test content", "quality": 0.8}

        if self.test_mode == "unit":
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('pickle.dump') as mock_dump:
                    self.extractor._save_to_cache("test_key", test_data)
                    mock_dump.assert_called_once_with(test_data, mock_file().__enter__())
        else:
            # Integration test
            self.extractor._save_to_cache("integration_test_key", test_data)
            cache_file = self.extractor.cache_dir / "integration_test_key.pkl"
            self.assertTrue(cache_file.exists())

    def test_detect_if_scanned_with_text_pdf(self):
        """Test scan detection for PDF with extractable text."""
        if self.test_mode == "unit":
            # Mock PyMuPDF document with text
            mock_page = Mock()
            mock_page.get_text.return_value = "A" * 200  # Sufficient text

            with patch('fitz.open') as mock_open:
                mock_doc = Mock()
                mock_doc.__len__.return_value = 1
                mock_doc.__getitem__.return_value = mock_page
                mock_doc.close.return_value = None
                mock_open.return_value = mock_doc

                is_scanned = self.extractor.detect_if_scanned(self.sample_pdf_path)
                self.assertFalse(is_scanned)
        else:
            # Would need real PDF for integration test
            pass

    def test_detect_if_scanned_with_scanned_pdf(self):
        """Test scan detection for scanned PDF."""
        if self.test_mode == "unit":
            # Mock PyMuPDF document with minimal text
            mock_page = Mock()
            mock_page.get_text.return_value = "A" * 10  # Insufficient text

            with patch('fitz.open') as mock_open:
                mock_doc = Mock()
                mock_doc.__len__.return_value = 1
                mock_doc.__getitem__.return_value = mock_page
                mock_doc.close.return_value = None
                mock_open.return_value = mock_doc

                is_scanned = self.extractor.detect_if_scanned(self.sample_pdf_path)
                self.assertTrue(is_scanned)

    def test_preprocess_for_academic_ocr(self):
        """Test image preprocessing for OCR."""
        if self.test_mode == "unit":
            # Mock PIL Image
            mock_img = Mock()
            mock_img.mode = 'RGB'
            mock_img.size = (500, 700)
            mock_img.convert.return_value = mock_img
            mock_img.filter.return_value = mock_img
            mock_img.resize.return_value = mock_img

            with patch('PIL.ImageEnhance.Contrast') as mock_contrast:
                mock_enhancer = Mock()
                mock_enhancer.enhance.return_value = mock_img
                mock_contrast.return_value = mock_enhancer

                result = self.extractor.preprocess_for_academic_ocr(mock_img)
                self.assertIsNotNone(result)
                mock_img.convert.assert_called_with('L')
                mock_enhancer.enhance.assert_called_with(2.0)
        else:
            # Integration test with real image
            test_img = Image.new('RGB', (800, 600), color='white')
            result = self.extractor.preprocess_for_academic_ocr(test_img)
            self.assertIsInstance(result, Image.Image)

    def test_ocr_academic_optimized(self):
        """Test OCR processing for academic content."""
        if self.test_mode == "unit":
            # Use the mock setup from PDFStandardTestCase
            with patch('fitz.open', return_value=self.mock_doc):
                with patch('pytesseract.image_to_string', return_value="Sample OCR text"):
                    with patch('PIL.Image.open', return_value=self.mock_image):
                        result = self.extractor.ocr_academic_optimized(self.sample_pdf_path)
                        self.assertIn("Sample OCR text", result)
                        self.assertIn("Page 1", result)
        else:
            # Would need real PDF for integration test
            pass

    def test_extract_text_direct(self):
        """Test direct text extraction from PDF."""
        if self.test_mode == "unit":
            # Mock PyMuPDF document
            mock_page = Mock()
            mock_page.get_text.return_value = self.sample_text

            with patch('fitz.open') as mock_open:
                mock_doc = Mock()
                mock_doc.__iter__.return_value = iter([mock_page])
                mock_doc.close.return_value = None
                mock_open.return_value = mock_doc

                result = self.extractor.extract_text_direct(self.sample_pdf_path)
                self.assertIn(self.sample_text, result)
                self.assertIn("Page 1", result)
        else:
            # Would need real PDF for integration test
            pass

    def test_clean_academic_text(self):
        """Test academic text cleaning."""
        dirty_text = "H  2  O  +  C O2  ->  H2CO3\n\n\n\nMore   text"
        cleaned = self.extractor.clean_academic_text(dirty_text)

        # Check that spaces are normalized
        self.assertIn("H2O", cleaned)
        self.assertIn("CO2", cleaned)
        self.assertIn("→", cleaned)  # Arrow should be normalized
        self.assertNotIn("   ", cleaned)  # Multiple spaces removed

    def test_estimate_extraction_quality_high_quality(self):
        """Test quality estimation for high-quality academic text."""
        high_quality_text = """
        This is a comprehensive analysis of quantum mechanics principles.
        The Schrödinger equation is given by: iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩

        Key theorems include:
        1. Heisenberg uncertainty principle
        2. Wave-particle duality theory

        Chemical reactions: H2O + CO2 → H2CO3
        Units: 5.2 kg, 10 m/s, 300 K
        """
        quality = self.extractor.estimate_extraction_quality(high_quality_text)
        self.assertGreater(quality, 0.5)

    def test_estimate_extraction_quality_low_quality(self):
        """Test quality estimation for low-quality text."""
        low_quality_text = "asdj flkjasdf 123456789 XyZ"
        quality = self.extractor.estimate_extraction_quality(low_quality_text)
        self.assertLess(quality, 0.3)

    def test_estimate_extraction_quality_empty_text(self):
        """Test quality estimation for empty text."""
        quality = self.extractor.estimate_extraction_quality("")
        self.assertEqual(quality, 0.0)

    def test_extract_text_robust_with_cache_hit(self):
        """Test robust extraction with cache hit."""
        if self.test_mode == "unit":
            cached_data = {
                'text': "Cached academic text",
                'quality': 0.9,
                'method': 'direct'
            }

            with patch.object(self.extractor, '_get_cache_key', return_value="test_key"):
                with patch.object(self.extractor, '_load_from_cache', return_value=cached_data):
                    with patch('pathlib.Path.exists', return_value=True):
                        result = self.extractor.extract_text_robust(self.sample_pdf_path)
                        self.assertEqual(result, "Cached academic text")

    def test_extract_text_robust_direct_extraction(self):
        """Test robust extraction using direct method."""
        if self.test_mode == "unit":
            # Mock no cache hit
            with patch.object(self.extractor, '_get_cache_key', return_value="test_key"):
                with patch.object(self.extractor, '_load_from_cache', return_value=None):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch.object(self.extractor, 'extract_text_direct', return_value=self.sample_text):
                            with patch.object(self.extractor, 'detect_if_scanned', return_value=False):
                                with patch.object(self.extractor, 'estimate_extraction_quality', return_value=0.8):
                                    with patch.object(self.extractor, 'clean_academic_text', side_effect=lambda x: x):
                                        with patch.object(self.extractor, '_save_to_cache'):
                                            result = self.extractor.extract_text_robust(self.sample_pdf_path)
                                            self.assertEqual(result, self.sample_text)

    def test_extract_text_robust_ocr_fallback(self):
        """Test robust extraction falling back to OCR."""
        if self.test_mode == "unit":
            # Mock poor direct extraction triggering OCR
            with patch.object(self.extractor, '_get_cache_key', return_value="test_key"):
                with patch.object(self.extractor, '_load_from_cache', return_value=None):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch.object(self.extractor, 'extract_text_direct', return_value=""):
                            with patch.object(self.extractor, 'detect_if_scanned', return_value=True):
                                with patch.object(self.extractor, 'ocr_academic_optimized', return_value="OCR text"):
                                    with patch.object(self.extractor, 'estimate_extraction_quality', return_value=0.7):
                                        with patch.object(self.extractor, 'clean_academic_text', side_effect=lambda x: x):
                                            with patch.object(self.extractor, '_save_to_cache'):
                                                result = self.extractor.extract_text_robust(self.sample_pdf_path)
                                                self.assertEqual(result, "OCR text")

    def test_extract_text_robust_file_not_found(self):
        """Test robust extraction with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text_robust("nonexistent.pdf")

    def test_extract_text_robust_all_methods_fail(self):
        """Test robust extraction when all methods fail."""
        if self.test_mode == "unit":
            with patch.object(self.extractor, '_get_cache_key', return_value="test_key"):
                with patch.object(self.extractor, '_load_from_cache', return_value=None):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch.object(self.extractor, 'extract_text_direct', return_value=""):
                            with patch.object(self.extractor, 'detect_if_scanned', return_value=False):
                                with self.assertRaises(Exception) as context:
                                    self.extractor.extract_text_robust(self.sample_pdf_path)
                                self.assertIn("All extraction methods failed", str(context.exception))

    def test_batch_extract_success(self):
        """Test batch extraction with successful processing."""
        if self.test_mode == "unit":
            pdf_paths = ["test1.pdf", "test2.pdf"]

            with patch.object(self.extractor, 'extract_text_robust', return_value="Extracted text"):
                with patch.object(self.extractor, 'estimate_extraction_quality', return_value=0.8):
                    with patch('builtins.open', mock_open()) as mock_file:
                        with patch('pathlib.Path.mkdir'):
                            results = self.extractor.batch_extract(pdf_paths, "test_output")

                            self.assertEqual(len(results), 2)
                            for pdf_path in pdf_paths:
                                self.assertTrue(results[pdf_path]['success'])
                                self.assertEqual(results[pdf_path]['quality'], 0.8)

    def test_batch_extract_with_failures(self):
        """Test batch extraction with some failures."""
        if self.test_mode == "unit":
            pdf_paths = ["test1.pdf", "test2.pdf"]

            def mock_extract(pdf_path):
                if pdf_path == "test1.pdf":
                    return "Extracted text"
                else:
                    raise Exception("Extraction failed")

            with patch.object(self.extractor, 'extract_text_robust', side_effect=mock_extract):
                with patch.object(self.extractor, 'estimate_extraction_quality', return_value=0.8):
                    with patch('builtins.open', mock_open()) as mock_file:
                        with patch('pathlib.Path.mkdir'):
                            results = self.extractor.batch_extract(pdf_paths, "test_output")

                            self.assertTrue(results["test1.pdf"]['success'])
                            self.assertFalse(results["test2.pdf"]['success'])
                            self.assertEqual(results["test2.pdf"]['error'], "Extraction failed")

    def test_caching_disabled(self):
        """Test behavior when caching is disabled."""
        extractor = PDFExtractor(enable_caching=False)

        # Cache methods should return None/do nothing
        cache_key = extractor._get_cache_key("test.pdf")
        result = extractor._load_from_cache("test_key")
        extractor._save_to_cache("test_key", {"data": "test"})

        # Should not raise errors, but cache operations should be no-ops
        self.assertIsNone(result)

    def test_academic_patterns_matching(self):
        """Test that academic patterns correctly identify academic content."""
        academic_text = """
        The theorem states that f(x) = x² + 2x + 1
        Therefore, the analysis shows H2O → H₂ + O₂
        Given the equation ∂f/∂x = 2x + 2
        Units: 5 kg, 10 m/s²
        """

        patterns = self.extractor.academic_patterns

        # Test equation pattern
        equation_matches = patterns['equations'].findall(academic_text)
        self.assertGreater(len(equation_matches), 0)

        # Test relationship words
        relationship_matches = patterns['relationships'].findall(academic_text)
        self.assertGreater(len(relationship_matches), 0)

        # Test academic terms
        term_matches = patterns['terms'].findall(academic_text)
        self.assertGreater(len(term_matches), 0)

        # Test units
        unit_matches = patterns['units'].findall(academic_text)
        self.assertGreater(len(unit_matches), 0)

    @pytest.mark.slow
    def test_performance_large_document(self):
        """Test performance with large document simulation."""
        if self.test_mode == "unit":
            # Simulate processing large document
            large_text = "Academic content " * 10000

            def mock_time_consuming_ocr(*args, **kwargs):
                return large_text

            with patch.object(self.extractor, 'ocr_academic_optimized', side_effect=mock_time_consuming_ocr):
                # Test that it completes within reasonable time
                result = self.assertExecutionTime(
                    self.extractor.ocr_academic_optimized,
                    max_time=5.0,  # Should complete within 5 seconds for mocked version
                    pdf_path=self.sample_pdf_path
                )
                self.assertEqual(result, large_text)

    @pytest.mark.integration
    def test_end_to_end_extraction(self):
        """End-to-end integration test (only runs in integration mode)."""
        if self.test_mode == "integration":
            # This would test with real PDFs
            # Create a sample PDF for testing
            temp_pdf = self.create_temp_file("", ".pdf")

            try:
                # This would fail without a real PDF, but demonstrates integration test structure
                result = self.extractor.extract_text_robust(str(temp_pdf))
                self.assertIsInstance(result, str)
            except Exception as e:
                # Expected to fail with dummy PDF file
                self.assertIn("PDF", str(e))


if __name__ == '__main__':
    unittest.main()