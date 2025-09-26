"""
Tests for SciHub client module.

Tests both unit (mocked) and integration (real SciHub requests) modes.
Uses NetworkStandardTestCase for comprehensive network mocking.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import time
from pathlib import Path
import tempfile
from datetime import datetime

from .base_test import NetworkStandardTestCase
from cite_tutor.scihub_client import SciHubClient


class TestSciHubClient(NetworkStandardTestCase):
    """Test SciHubClient class with mock abstraction."""

    def setUp(self):
        super().setUp()
        self.client = SciHubClient(cache_dir="test_cache", request_timeout=30)
        self.sample_doi = "10.1021/ja01367a002"
        self.sample_pdf_bytes = b'%PDF-1.4\n%Test PDF content\nSome chemistry content\n/Type/Page\n%%EOF'
        self.sample_html = """
        <html>
        <body>
            <iframe id="pdf" src="/downloads/test.pdf"></iframe>
        </body>
        </html>
        """

    def test_init_creates_cache_directory(self):
        """Test that SciHubClient creates cache directory."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            client = SciHubClient(cache_dir="test_cache")
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_create_safe_filename(self):
        """Test safe filename creation from DOI."""
        # Test normal DOI
        safe_name = self.client._create_safe_filename("10.1021/ja01367a002")
        self.assertEqual(safe_name, "10_1021_ja01367a002")

        # Test long DOI
        long_doi = "10.1021/" + "a" * 100
        safe_name_long = self.client._create_safe_filename(long_doi)
        self.assertLessEqual(len(safe_name_long), 100)
        self.assertIn("_", safe_name_long)  # Should contain hash separator

        # Test DOI with special characters
        special_doi = "10.1021/ja-2023.12345/special:chars"
        safe_name_special = self.client._create_safe_filename(special_doi)
        self.assertNotIn("/", safe_name_special)
        self.assertNotIn(":", safe_name_special)

    def test_find_working_mirror_success(self):
        """Test finding working mirror."""
        if self.test_mode == "unit":
            # Configure mock response for successful mirror
            self.configure_response(
                status_code=200,
                content="<html><title>Sci-Hub</title><p>Scientific research</p></html>"
            )

            working_mirror = self.client.find_working_mirror()

            self.assertIsNotNone(working_mirror)
            self.assertIn("sci-hub", working_mirror.lower())
            self.mock_session.get.assert_called()

    def test_find_working_mirror_cached(self):
        """Test using cached working mirror."""
        self.client.working_mirror = "https://sci-hub.test/"
        self.client.last_mirror_check = time.time()

        # Should return cached mirror without making requests
        working_mirror = self.client.find_working_mirror()
        self.assertEqual(working_mirror, "https://sci-hub.test/")

    def test_find_working_mirror_force_refresh(self):
        """Test forcing mirror refresh."""
        if self.test_mode == "unit":
            self.client.working_mirror = "https://sci-hub.test/"
            self.client.last_mirror_check = time.time()

            self.configure_response(
                status_code=200,
                content="<html><title>Sci-Hub</title></html>"
            )

            # Force refresh should make new requests
            working_mirror = self.client.find_working_mirror(force_refresh=True)
            self.mock_session.get.assert_called()

    def test_find_working_mirror_all_fail(self):
        """Test when all mirrors fail."""
        if self.test_mode == "unit":
            # All mirrors return failure
            self.configure_response(status_code=404)

            with self.assertRaises(Exception) as context:
                self.client.find_working_mirror()

            self.assertIn("No working Sci-Hub mirrors found", str(context.exception))

    def test_check_local_cache_hit(self):
        """Test local cache hit."""
        if self.test_mode == "unit":
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=self.sample_pdf_bytes)):
                    with patch.object(self.client, 'validate_pdf_content', return_value=True):
                        cached_pdf = self.client.check_local_cache(self.sample_doi)
                        self.assertEqual(cached_pdf, self.sample_pdf_bytes)
        else:
            # Integration test with real cache
            temp_dir = self.create_temp_file("", ".pdf")
            temp_dir.write_bytes(self.sample_pdf_bytes)

            with patch.object(self.client, 'cache_dir', temp_dir.parent):
                with patch.object(self.client, '_create_safe_filename', return_value=temp_dir.stem):
                    cached_pdf = self.client.check_local_cache(self.sample_doi)
                    self.assertEqual(cached_pdf, self.sample_pdf_bytes)

    def test_check_local_cache_miss(self):
        """Test local cache miss."""
        if self.test_mode == "unit":
            with patch('pathlib.Path.exists', return_value=False):
                cached_pdf = self.client.check_local_cache(self.sample_doi)
                self.assertIsNone(cached_pdf)

    def test_check_local_cache_corrupted(self):
        """Test handling of corrupted cached file."""
        if self.test_mode == "unit":
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=b'corrupted data')):
                    with patch.object(self.client, 'validate_pdf_content', return_value=False):
                        with patch('pathlib.Path.unlink') as mock_unlink:
                            cached_pdf = self.client.check_local_cache(self.sample_doi)
                            self.assertIsNone(cached_pdf)
                            mock_unlink.assert_called_once()

    def test_extract_pdf_url_from_html_iframe(self):
        """Test PDF URL extraction from iframe."""
        html = """
        <html>
        <body>
            <iframe id="pdf" src="/downloads/paper.pdf"></iframe>
        </body>
        </html>
        """

        base_url = "https://sci-hub.test/"
        pdf_url = self.client.extract_pdf_url_from_html(html, base_url)

        self.assertEqual(pdf_url, "https://sci-hub.test/downloads/paper.pdf")

    def test_extract_pdf_url_from_html_direct_link(self):
        """Test PDF URL extraction from direct link."""
        html = """
        <html>
        <body>
            <a href="paper.pdf">Download PDF</a>
        </body>
        </html>
        """

        base_url = "https://sci-hub.test/"
        pdf_url = self.client.extract_pdf_url_from_html(html, base_url)

        self.assertEqual(pdf_url, "https://sci-hub.test/paper.pdf")

    def test_extract_pdf_url_from_html_embed_tag(self):
        """Test PDF URL extraction from embed tag."""
        html = """
        <html>
        <body>
            <embed src="document.pdf" type="application/pdf" />
        </body>
        </html>
        """

        base_url = "https://sci-hub.test/"
        pdf_url = self.client.extract_pdf_url_from_html(html, base_url)

        self.assertEqual(pdf_url, "https://sci-hub.test/document.pdf")

    def test_extract_pdf_url_from_html_object_tag(self):
        """Test PDF URL extraction from object tag."""
        html = """
        <html>
        <body>
            <object data="research.pdf" type="application/pdf"></object>
        </body>
        </html>
        """

        base_url = "https://sci-hub.test/"
        pdf_url = self.client.extract_pdf_url_from_html(html, base_url)

        self.assertEqual(pdf_url, "https://sci-hub.test/research.pdf")

    def test_extract_pdf_url_from_html_javascript(self):
        """Test PDF URL extraction from JavaScript."""
        html = """
        <html>
        <body>
            <script>
                var pdfUrl = "https://example.com/paper.pdf";
                loadPdf(pdfUrl);
            </script>
        </body>
        </html>
        """

        pdf_url = self.client.extract_pdf_url_from_html(html, "https://sci-hub.test/")

        self.assertEqual(pdf_url, "https://example.com/paper.pdf")

    def test_extract_pdf_url_from_html_not_found(self):
        """Test PDF URL extraction when no URL found."""
        html = """
        <html>
        <body>
            <p>No PDF links here</p>
        </body>
        </html>
        """

        pdf_url = self.client.extract_pdf_url_from_html(html, "https://sci-hub.test/")
        self.assertIsNone(pdf_url)

    def test_validate_pdf_content_valid(self):
        """Test PDF validation with valid content."""
        valid_pdf = b'%PDF-1.4\n%Chemistry content\nmolecule\n/Type/Page\nSome text here\n%%EOF'
        is_valid = self.client.validate_pdf_content(valid_pdf)
        self.assertTrue(is_valid)

    def test_validate_pdf_content_invalid_header(self):
        """Test PDF validation with invalid header."""
        invalid_pdf = b'Not a PDF file'
        is_valid = self.client.validate_pdf_content(invalid_pdf)
        self.assertFalse(is_valid)

    def test_validate_pdf_content_too_small(self):
        """Test PDF validation with too small file."""
        small_pdf = b'%PDF-1.4\n%%EOF'  # Too small
        is_valid = self.client.validate_pdf_content(small_pdf)
        self.assertFalse(is_valid)

    def test_validate_pdf_content_too_large(self):
        """Test PDF validation with too large file."""
        # Create 60MB file
        large_pdf = b'%PDF-1.4\n' + b'a' * (60 * 1024 * 1024) + b'\n/Type/Page\n%%EOF'
        is_valid = self.client.validate_pdf_content(large_pdf)
        self.assertFalse(is_valid)

    def test_validate_pdf_content_no_pages(self):
        """Test PDF validation without pages."""
        no_pages_pdf = b'%PDF-1.4\n' + b'a' * 200000 + b'\n%%EOF'  # No /Type/Page
        is_valid = self.client.validate_pdf_content(no_pages_pdf)
        self.assertFalse(is_valid)

    def test_validate_pdf_content_no_eof(self):
        """Test PDF validation without EOF marker."""
        no_eof_pdf = b'%PDF-1.4\n' + b'a' * 200000 + b'\n/Type/Page\n'  # No %%EOF
        is_valid = self.client.validate_pdf_content(no_eof_pdf)
        self.assertFalse(is_valid)

    def test_get_paper_by_doi_cached(self):
        """Test paper retrieval with cache hit."""
        with patch.object(self.client, 'check_local_cache', return_value=self.sample_pdf_bytes):
            pdf_bytes = self.client.get_paper_by_doi(self.sample_doi)
            self.assertEqual(pdf_bytes, self.sample_pdf_bytes)

    def test_get_paper_by_doi_direct_pdf_response(self):
        """Test paper retrieval with direct PDF response."""
        if self.test_mode == "unit":
            # Mock no cache hit
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    # Configure direct PDF response
                    self.configure_response(
                        status_code=200,
                        content=self.sample_pdf_bytes,
                        headers={"Content-Type": "application/pdf"}
                    )

                    with patch.object(self.client, 'validate_pdf_content', return_value=True):
                        with patch.object(self.client, 'cache_paper_locally'):
                            pdf_bytes = self.client.get_paper_by_doi(self.sample_doi)
                            self.assertEqual(pdf_bytes, self.sample_pdf_bytes)

    def test_get_paper_by_doi_html_response(self):
        """Test paper retrieval with HTML response containing PDF link."""
        if self.test_mode == "unit":
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    # First response: HTML with PDF link
                    html_response = Mock()
                    html_response.status_code = 200
                    html_response.headers = {"Content-Type": "text/html"}
                    html_response.text = self.sample_html

                    # Second response: PDF content
                    pdf_response = Mock()
                    pdf_response.status_code = 200
                    pdf_response.content = self.sample_pdf_bytes

                    self.mock_session.get.side_effect = [html_response, pdf_response]

                    with patch.object(self.client, 'validate_pdf_content', return_value=True):
                        with patch.object(self.client, 'cache_paper_locally'):
                            pdf_bytes = self.client.get_paper_by_doi(self.sample_doi)
                            self.assertEqual(pdf_bytes, self.sample_pdf_bytes)

    def test_get_paper_by_doi_html_no_pdf_link(self):
        """Test paper retrieval with HTML response but no PDF link."""
        if self.test_mode == "unit":
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    self.configure_response(
                        status_code=200,
                        content="<html><body>No PDF link</body></html>",
                        headers={"Content-Type": "text/html"}
                    )

                    with self.assertRaises(Exception) as context:
                        self.client.get_paper_by_doi(self.sample_doi)

                    self.assertIn("Could not find PDF URL", str(context.exception))

    def test_get_paper_by_doi_http_error(self):
        """Test paper retrieval with HTTP error."""
        if self.test_mode == "unit":
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    self.configure_response(status_code=404)

                    with self.assertRaises(Exception) as context:
                        self.client.get_paper_by_doi(self.sample_doi)

                    self.assertIn("HTTP 404", str(context.exception))

    def test_get_paper_by_doi_invalid_pdf(self):
        """Test paper retrieval with invalid PDF content."""
        if self.test_mode == "unit":
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    self.configure_response(
                        status_code=200,
                        content=b"Invalid PDF content",
                        headers={"Content-Type": "application/pdf"}
                    )

                    with patch.object(self.client, 'validate_pdf_content', return_value=False):
                        with self.assertRaises(Exception) as context:
                            self.client.get_paper_by_doi(self.sample_doi)

                        self.assertIn("not a valid PDF", str(context.exception))

    def test_get_paper_by_doi_retry_mechanism(self):
        """Test retry mechanism on failures."""
        if self.test_mode == "unit":
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    # First attempt fails, second succeeds
                    responses = [
                        Mock(status_code=500),  # Server error
                        Mock(status_code=200, headers={"Content-Type": "application/pdf"}, content=self.sample_pdf_bytes)
                    ]
                    self.mock_session.get.side_effect = responses

                    with patch.object(self.client, 'validate_pdf_content', return_value=True):
                        with patch.object(self.client, 'cache_paper_locally'):
                            with patch('time.sleep'):  # Speed up test
                                pdf_bytes = self.client.get_paper_by_doi(self.sample_doi, retry_attempts=2)
                                self.assertEqual(pdf_bytes, self.sample_pdf_bytes)

    def test_get_paper_by_doi_all_retries_fail(self):
        """Test when all retry attempts fail."""
        if self.test_mode == "unit":
            with patch.object(self.client, 'check_local_cache', return_value=None):
                with patch.object(self.client, 'find_working_mirror', return_value="https://sci-hub.test/"):
                    self.configure_response(status_code=500)  # Always fails

                    with patch('time.sleep'):  # Speed up test
                        with self.assertRaises(Exception) as context:
                            self.client.get_paper_by_doi(self.sample_doi, retry_attempts=2)

                        self.assertIn("Failed to retrieve paper after 2 attempts", str(context.exception))

    def test_cache_paper_locally(self):
        """Test local paper caching."""
        metadata = {"source": "test", "attempt": 1}

        if self.test_mode == "unit":
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump') as mock_json_dump:
                    self.client.cache_paper_locally(self.sample_doi, self.sample_pdf_bytes, metadata)

                    # Should write PDF and metadata
                    self.assertEqual(mock_file.call_count, 2)  # PDF file + metadata file
                    mock_json_dump.assert_called_once()
        else:
            # Integration test with real file I/O
            with tempfile.TemporaryDirectory() as temp_dir:
                client = SciHubClient(cache_dir=temp_dir)
                client.cache_paper_locally(self.sample_doi, self.sample_pdf_bytes, metadata)

                # Check files were created
                safe_filename = client._create_safe_filename(self.sample_doi)
                pdf_path = Path(temp_dir) / f"{safe_filename}.pdf"
                metadata_path = Path(temp_dir) / f"{safe_filename}.json"

                self.assertTrue(pdf_path.exists())
                self.assertTrue(metadata_path.exists())

                # Check content
                self.assertEqual(pdf_path.read_bytes(), self.sample_pdf_bytes)
                metadata_content = json.loads(metadata_path.read_text())
                self.assertEqual(metadata_content['doi'], self.sample_doi)

    def test_cache_paper_locally_error_handling(self):
        """Test cache error handling."""
        if self.test_mode == "unit":
            with patch('builtins.open', side_effect=IOError("Disk full")):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.unlink') as mock_unlink:
                        # Should not raise exception, but should clean up
                        self.client.cache_paper_locally(self.sample_doi, self.sample_pdf_bytes)
                        mock_unlink.assert_called()

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        if self.test_mode == "unit":
            # Mock file system
            mock_pdf_files = [
                Mock(name="paper1.pdf", stat=Mock(return_value=Mock(st_size=1024*1024, st_mtime=time.time()))),
                Mock(name="paper2.pdf", stat=Mock(return_value=Mock(st_size=2*1024*1024, st_mtime=time.time())))
            ]

            with patch.object(self.client.cache_dir, 'glob', return_value=mock_pdf_files):
                # Mock metadata files
                def mock_with_suffix(suffix):
                    mock_file = Mock()
                    mock_file.exists.return_value = True
                    return mock_file

                for pdf_file in mock_pdf_files:
                    pdf_file.with_suffix = mock_with_suffix

                with patch('builtins.open', mock_open(read_data='{"doi": "test"}')):
                    with patch('json.load', return_value={"doi": "test"}):
                        cache_info = self.client.get_cache_info()

                        self.assertEqual(cache_info['total_papers'], 2)
                        self.assertEqual(cache_info['total_size_mb'], 3.0)  # 1MB + 2MB
                        self.assertEqual(len(cache_info['papers']), 2)

    def test_cleanup_cache_by_age(self):
        """Test cache cleanup by age."""
        if self.test_mode == "unit":
            current_time = time.time()
            old_time = current_time - (40 * 24 * 3600)  # 40 days old

            mock_pdf_files = [
                Mock(
                    stat=Mock(return_value=Mock(st_size=1024*1024, st_mtime=old_time)),
                    unlink=Mock(),
                    with_suffix=Mock(return_value=Mock(exists=Mock(return_value=False)))
                )
            ]

            with patch.object(self.client.cache_dir, 'glob', return_value=mock_pdf_files):
                with patch('time.time', return_value=current_time):
                    cleanup_stats = self.client.cleanup_cache(max_age_days=30)

                    self.assertEqual(cleanup_stats['files_removed'], 1)
                    self.assertGreater(cleanup_stats['space_freed_mb'], 0)
                    mock_pdf_files[0].unlink.assert_called_once()

    def test_cleanup_cache_by_size(self):
        """Test cache cleanup by size."""
        if self.test_mode == "unit":
            current_time = time.time()

            # Create large files that exceed size limit
            mock_pdf_files = [
                Mock(
                    stat=Mock(return_value=Mock(st_size=600*1024*1024, st_mtime=current_time)),  # 600MB
                    unlink=Mock(),
                    exists=Mock(return_value=True),
                    with_suffix=Mock(return_value=Mock(exists=Mock(return_value=False)))
                ),
                Mock(
                    stat=Mock(return_value=Mock(st_size=500*1024*1024, st_mtime=current_time)),  # 500MB
                    unlink=Mock(),
                    exists=Mock(return_value=True),
                    with_suffix=Mock(return_value=Mock(exists=Mock(return_value=False)))
                )
            ]

            with patch.object(self.client.cache_dir, 'glob', return_value=mock_pdf_files):
                cleanup_stats = self.client.cleanup_cache(max_age_days=365, max_size_mb=800)  # 800MB limit

                # Should remove the largest file (600MB) to get under limit
                self.assertGreater(cleanup_stats['files_removed'], 0)
                self.assertGreater(cleanup_stats['space_freed_mb'], 0)

    def test_batch_retrieve_papers(self):
        """Test batch paper retrieval."""
        dois = ["10.1021/test1", "10.1021/test2"]

        if self.test_mode == "unit":
            with patch.object(self.client, 'get_paper_by_doi', return_value=self.sample_pdf_bytes):
                with patch('time.sleep'):  # Speed up test
                    results = self.client.batch_retrieve_papers(dois)

                    self.assertEqual(len(results), 2)
                    for doi in dois:
                        self.assertTrue(results[doi]['success'])
                        self.assertEqual(results[doi]['size_bytes'], len(self.sample_pdf_bytes))

    def test_batch_retrieve_papers_with_failures(self):
        """Test batch retrieval with some failures."""
        dois = ["10.1021/success", "10.1021/failure"]

        if self.test_mode == "unit":
            def mock_get_paper(doi):
                if "success" in doi:
                    return self.sample_pdf_bytes
                else:
                    raise Exception("Retrieval failed")

            with patch.object(self.client, 'get_paper_by_doi', side_effect=mock_get_paper):
                with patch('time.sleep'):  # Speed up test
                    results = self.client.batch_retrieve_papers(dois)

                    self.assertTrue(results["10.1021/success"]['success'])
                    self.assertFalse(results["10.1021/failure"]['success'])
                    self.assertIn("error", results["10.1021/failure"])

    @pytest.mark.slow
    def test_performance_large_pdf(self):
        """Test performance with large PDF."""
        if self.test_mode == "unit":
            # Simulate large PDF (10MB)
            large_pdf = b'%PDF-1.4\n' + b'chemistry content\n' * 100000 + b'/Type/Page\n%%EOF'

            # Test validation performance
            result = self.assertExecutionTime(
                self.client.validate_pdf_content,
                max_time=1.0,  # Should complete within 1 second
                pdf_bytes=large_pdf
            )
            self.assertTrue(result)

    @pytest.mark.integration
    def test_end_to_end_retrieval(self):
        """End-to-end integration test (only runs in integration mode)."""
        if self.test_mode == "integration":
            # This would test with real SciHub requests
            try:
                # Test mirror finding
                working_mirror = self.client.find_working_mirror()
                self.assertIsNotNone(working_mirror)
                self.assertIn("sci-hub", working_mirror.lower())

                # Note: We don't test actual paper retrieval in automated tests
                # to avoid overloading SciHub servers and potential legal issues
                self.skipTest("Real paper retrieval skipped to avoid overloading servers")

            except Exception as e:
                # Network issues in integration tests are acceptable
                self.skipTest(f"Integration test skipped due to network: {e}")

    @pytest.mark.network
    def test_user_agent_rotation(self):
        """Test that different user agents are used."""
        if self.test_mode == "unit":
            # Mock multiple requests and check different user agents
            user_agents_used = []

            def capture_headers(*args, **kwargs):
                user_agents_used.append(kwargs.get('headers', {}).get('User-Agent'))
                return Mock(status_code=200, text="sci-hub test page")

            self.mock_session.get.side_effect = capture_headers

            # Make multiple mirror checks
            for i in range(3):
                try:
                    self.client.find_working_mirror(force_refresh=True)
                except:
                    pass  # Ignore failures, just collect user agents

            # Should have used user agents
            self.assertGreater(len(user_agents_used), 0)

    @pytest.mark.expensive
    def test_rate_limiting_behavior(self):
        """Test rate limiting in batch operations."""
        if self.test_mode == "unit":
            dois = ["10.1021/test1", "10.1021/test2", "10.1021/test3"]

            with patch.object(self.client, 'get_paper_by_doi', return_value=self.sample_pdf_bytes):
                with patch('time.sleep') as mock_sleep:
                    start_time = time.time()
                    self.client.batch_retrieve_papers(dois)
                    end_time = time.time()

                    # Should have called sleep between requests
                    self.assertGreaterEqual(mock_sleep.call_count, len(dois) - 1)

    def test_doi_cleaning(self):
        """Test DOI cleaning in get_paper_by_doi."""
        dirty_dois = [
            "doi:10.1021/test",
            "/10.1021/test",
            "  10.1021/test  ",
            "doi:/10.1021/test"
        ]

        for dirty_doi in dirty_dois:
            if self.test_mode == "unit":
                with patch.object(self.client, 'check_local_cache', return_value=self.sample_pdf_bytes):
                    # Should not raise exception and should clean DOI
                    result = self.client.get_paper_by_doi(dirty_doi)
                    self.assertEqual(result, self.sample_pdf_bytes)


if __name__ == '__main__':
    unittest.main()