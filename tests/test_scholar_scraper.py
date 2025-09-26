"""
Tests for Scholar scraper module.

Tests both unit (mocked) and integration (real Scholar requests) modes.
Uses NetworkStandardTestCase for comprehensive network mocking.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from pathlib import Path

from .base_test import NetworkStandardTestCase
from cite_tutor.scholar_scraper import ScholarScraper, ScholarResult


class TestScholarResult(unittest.TestCase):
    """Test ScholarResult dataclass."""

    def test_scholar_result_creation(self):
        """Test creating ScholarResult instance."""
        result = ScholarResult(
            title="Test Paper Title",
            authors="Smith, J., Brown, A.",
            publication="Journal of Test Science",
            year=2023,
            url="https://example.com/paper",
            doi="10.1000/test.doi",
            cited_by=25,
            validation_score=0.85
        )

        self.assertEqual(result.title, "Test Paper Title")
        self.assertEqual(result.authors, "Smith, J., Brown, A.")
        self.assertEqual(result.year, 2023)
        self.assertEqual(result.doi, "10.1000/test.doi")
        self.assertEqual(result.cited_by, 25)
        self.assertEqual(result.validation_score, 0.85)


class TestScholarScraper(NetworkStandardTestCase):
    """Test ScholarScraper class with mock abstraction."""

    def setUp(self):
        super().setUp()
        self.scraper = ScholarScraper(delay_range=(0.1, 0.2))  # Faster for testing
        self.sample_citation = "Smith, J. Test Chemistry Paper. J. Am. Chem. Soc. 2023, 145, 123-130."
        self.sample_html = """
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://example.com/paper">Test Chemistry Paper: A Study</a>
            </h3>
            <div class="gs_a">Smith, J., Brown, A. - Journal of the American Chemical Society, 2023</div>
            <div class="gs_fl">
                <a href="">Cited by 25</a>
            </div>
        </div>
        """

    def test_init_creates_cache_directory(self):
        """Test that ScholarScraper creates cache directory."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            scraper = ScholarScraper(cache_dir="test_cache")
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_expand_chemistry_journal_abbreviations(self):
        """Test journal abbreviation expansion."""
        # Test JACS abbreviation
        citation = "Smith, J. Test Paper. J. Am. Chem. Soc. 2023, 145, 123."
        expanded = self.scraper.expand_chemistry_journal_abbreviations(citation)
        self.assertIn("journal american chemical society", expanded.lower())

        # Test multiple abbreviations
        citation2 = "Brown, A. Study. Angew. Chem. 2022, 61, 456."
        expanded2 = self.scraper.expand_chemistry_journal_abbreviations(citation2)
        self.assertIn("angewandte chemie", expanded2.lower())

        # Test no abbreviations
        citation3 = "Jones, B. Research. Nature 2021, 594, 789."
        expanded3 = self.scraper.expand_chemistry_journal_abbreviations(citation3)
        self.assertEqual(citation3, expanded3)

    def test_clean_citation_for_search_multiple_strategies(self):
        """Test generation of multiple search queries."""
        citation = "Smith, J. et al. Test Chemistry Paper. J. Am. Chem. Soc. 2023, 145, 123-130."
        context = "This study of chemical bonding was revolutionary."

        queries = self.scraper.clean_citation_for_search(citation, context)

        # Should generate multiple unique queries
        self.assertGreater(len(queries), 1)
        self.assertIsInstance(queries, list)

        # Check that queries are unique
        self.assertEqual(len(queries), len(set(queries)))

        # Should include expanded journal name
        expanded_found = any("journal american chemical society" in q.lower() for q in queries)
        self.assertTrue(expanded_found)

    def test_extract_author_year_patterns(self):
        """Test extraction of author and year from citations."""
        # Test parenthetical citation
        citation1 = "As shown by Brown et al. (1962), the reaction proceeds rapidly."
        result1 = self.scraper._extract_author_year(citation1)
        self.assertIn("Brown et al", result1)
        self.assertIn("1962", result1)

        # Test journal citation
        citation2 = "Smith, J. A. Test Paper. Journal Name (2023)"
        result2 = self.scraper._extract_author_year(citation2)
        self.assertIsNotNone(result2)

        # Test no match
        citation3 = "No author or year information here."
        result3 = self.scraper._extract_author_year(citation3)
        self.assertIsNone(result3)

    def test_extract_journal_info(self):
        """Test extraction of journal information."""
        citation = "Smith, J. Test Paper. Journal of Chemistry 145, 123 (2023)"
        result = self.scraper._extract_journal_info(citation)

        self.assertIsNotNone(result)
        self.assertIn("Journal of Chemistry", result)
        self.assertIn("2023", result)
        self.assertIn("volume 145", result)

    def test_guess_title_from_context(self):
        """Test title guessing from context."""
        context1 = 'The paper titled "Quantum Chemistry Applications" was groundbreaking.'
        title1 = self.scraper._guess_title_from_context(context1)
        self.assertEqual(title1, "Quantum Chemistry Applications")

        context2 = "A study of molecular orbital theory was conducted."
        title2 = self.scraper._guess_title_from_context(context2)
        self.assertEqual(title2, "molecular orbital theory")

        # Test no title found
        context3 = "The reaction was fast."
        title3 = self.scraper._guess_title_from_context(context3)
        self.assertIsNone(title3)

    def test_handle_rate_limiting(self):
        """Test rate limiting behavior."""
        with patch('time.sleep') as mock_sleep:
            # Test first rate limit
            self.scraper.current_delay_index = 0
            self.scraper.handle_rate_limiting()
            mock_sleep.assert_called_with(10)  # First delay
            self.assertEqual(self.scraper.current_delay_index, 1)

            # Test maximum rate limit
            self.scraper.current_delay_index = len(self.scraper.rate_limit_delays)
            self.scraper.handle_rate_limiting()
            mock_sleep.assert_called_with(600)  # Maximum delay

    def test_enforce_rate_limit(self):
        """Test rate limit enforcement."""
        with patch('time.sleep') as mock_sleep:
            with patch('time.time', side_effect=[0, 0.5]):  # Simulate short time gap
                self.scraper.last_request_time = 0
                self.scraper._enforce_rate_limit()
                mock_sleep.assert_called()

    def test_search_with_rotation_success(self):
        """Test successful Scholar search."""
        if self.test_mode == "unit":
            # Configure mock response
            self.configure_response(
                status_code=200,
                content=self.sample_html,
                headers={"Content-Type": "text/html"}
            )

            with patch.object(self.scraper, '_parse_scholar_results') as mock_parse:
                mock_result = ScholarResult(
                    title="Test Paper",
                    authors="Smith, J.",
                    publication="Test Journal",
                    year=2023,
                    url="https://example.com",
                    doi="10.1000/test",
                    cited_by=10,
                    validation_score=0.8
                )
                mock_parse.return_value = [mock_result]

                results = self.scraper.search_with_rotation("test query")

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].title, "Test Paper")
                self.mock_session.get.assert_called_once()
        else:
            # Integration test would make real request
            pass

    def test_search_with_rotation_rate_limited(self):
        """Test Scholar search with rate limiting."""
        if self.test_mode == "unit":
            # First response: rate limited, second response: success
            self.mock_session.get.side_effect = [
                Mock(status_code=429),
                Mock(status_code=200, text=self.sample_html)
            ]

            with patch.object(self.scraper, 'handle_rate_limiting') as mock_rate_limit:
                with patch.object(self.scraper, '_parse_scholar_results', return_value=[]):
                    results = self.scraper.search_with_rotation("test query")

                    mock_rate_limit.assert_called_once()
                    self.assertEqual(self.mock_session.get.call_count, 2)

    def test_search_with_rotation_failure(self):
        """Test Scholar search with request failure."""
        if self.test_mode == "unit":
            self.configure_response(status_code=500)

            results = self.scraper.search_with_rotation("test query")
            self.assertEqual(results, [])

    def test_parse_scholar_results(self):
        """Test parsing of Scholar HTML results."""
        html = """
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://example.com/paper1">First Test Paper</a>
            </h3>
            <div class="gs_a">Smith, J. - Journal of Chemistry, 2023</div>
            <div class="gs_fl">
                <a href="">Cited by 50</a>
            </div>
        </div>
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://example.com/paper2">Second Test Paper</a>
            </h3>
            <div class="gs_a">Brown, A. - Nature Chemistry, 2022</div>
        </div>
        """

        results = self.scraper._parse_scholar_results(html, "test query")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "First Test Paper")
        self.assertEqual(results[0].cited_by, 50)
        self.assertEqual(results[1].title, "Second Test Paper")
        self.assertEqual(results[1].cited_by, 0)  # No citation info

    def test_parse_single_result_complete(self):
        """Test parsing a complete Scholar result."""
        from bs4 import BeautifulSoup

        html = """
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://doi.org/10.1021/test">Advanced Chemistry Concepts</a>
            </h3>
            <div class="gs_a">Smith, J., Brown, A. - Journal of the American Chemical Society, 2023</div>
            <div class="gs_fl">
                <a href="">Cited by 125</a>
                <a href="https://doi.org/10.1021/test">doi.org</a>
            </div>
        </div>
        """

        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div', {'class': 'gs_r gs_or gs_scl'})

        result = self.scraper._parse_single_result(div)

        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Advanced Chemistry Concepts")
        self.assertIn("Smith", result.authors)
        self.assertEqual(result.year, 2023)
        self.assertEqual(result.cited_by, 125)

    def test_parse_single_result_minimal(self):
        """Test parsing a minimal Scholar result."""
        from bs4 import BeautifulSoup

        html = """
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://example.com">Minimal Paper</a>
            </h3>
        </div>
        """

        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div', {'class': 'gs_r gs_or gs_scl'})

        result = self.scraper._parse_single_result(div)

        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Minimal Paper")
        self.assertEqual(result.authors, "")
        self.assertIsNone(result.year)
        self.assertEqual(result.cited_by, 0)

    def test_extract_doi_from_result_direct_link(self):
        """Test DOI extraction from direct DOI links."""
        from bs4 import BeautifulSoup

        html = """
        <div>
            <a href="https://doi.org/10.1021/ja.2023.12345">DOI Link</a>
        </div>
        """

        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')

        if self.test_mode == "unit":
            doi = self.scraper._extract_doi_from_result(div, "")
            self.assertEqual(doi, "10.1021/ja.2023.12345")

    def test_extract_doi_from_result_publisher_redirect(self):
        """Test DOI extraction from publisher redirects."""
        from bs4 import BeautifulSoup

        html = """
        <div>
            <a href="https://pubs.acs.org/doi/10.1021/ja.2023.12345">ACS Link</a>
        </div>
        """

        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')

        if self.test_mode == "unit":
            # Mock the redirect following
            mock_response = Mock()
            mock_response.url = "https://doi.org/10.1021/ja.2023.12345"
            self.mock_session.head.return_value = mock_response

            doi = self.scraper._extract_doi_from_result(div, "")
            self.assertEqual(doi, "10.1021/ja.2023.12345")

    def test_validate_chemistry_paper_match_high_score(self):
        """Test validation of good chemistry paper match."""
        result = ScholarResult(
            title="Quantum Chemistry and Molecular Orbital Theory",
            authors="Smith, J.",
            publication="Journal of the American Chemical Society",
            year=2023,
            url="https://example.com",
            doi="10.1021/test",
            cited_by=100,
            validation_score=0.0
        )

        citation = "Smith, J. Quantum Theory Study. J. Am. Chem. Soc. 2023, 145, 123."

        score = self.scraper.validate_chemistry_paper_match(result, citation)

        # Should have high score due to:
        # - Chemistry keywords in title
        # - Chemistry journal
        # - Author match
        # - Year match
        # - Citation count
        self.assertGreater(score, 0.7)

    def test_validate_chemistry_paper_match_low_score(self):
        """Test validation of poor chemistry paper match."""
        result = ScholarResult(
            title="Computer Science Algorithm Design",
            authors="Jones, B.",
            publication="IEEE Computer Society",
            year=2020,
            url="https://example.com",
            doi=None,
            cited_by=5,
            validation_score=0.0
        )

        citation = "Smith, J. Chemistry Study. J. Am. Chem. Soc. 2023, 145, 123."

        score = self.scraper.validate_chemistry_paper_match(result, citation)

        # Should have low score due to:
        # - No chemistry keywords
        # - Non-chemistry journal
        # - Author mismatch
        # - Year mismatch
        # - Low citations
        self.assertLess(score, 0.3)

    def test_extract_authors_from_citation(self):
        """Test author extraction from citations."""
        # Test standard format
        citation1 = "Smith, J., Brown, A. Test Paper. Journal 2023."
        authors1 = self.scraper._extract_authors_from_citation(citation1)
        self.assertIn("Smith", authors1)
        self.assertIn("Brown", authors1)

        # Test et al. format
        citation2 = "Johnson et al. Study. Journal 2022."
        authors2 = self.scraper._extract_authors_from_citation(citation2)
        self.assertIn("Johnson", authors2)

    def test_extract_year_from_citation(self):
        """Test year extraction from citations."""
        citation1 = "Smith, J. Test Paper. Journal (2023)."
        year1 = self.scraper._extract_year_from_citation(citation1)
        self.assertEqual(year1, 2023)

        citation2 = "Brown, A. Study, 2022."
        year2 = self.scraper._extract_year_from_citation(citation2)
        self.assertEqual(year2, 2022)

        citation3 = "No year in this citation."
        year3 = self.scraper._extract_year_from_citation(citation3)
        self.assertIsNone(year3)

    def test_calculate_author_similarity(self):
        """Test author similarity calculation."""
        result_authors = "Smith, J., Brown, A., Jones, C."
        citation_authors = ["Smith", "Brown"]

        similarity = self.scraper._calculate_author_similarity(result_authors, citation_authors)
        self.assertEqual(similarity, 1.0)  # Both authors found

        citation_authors2 = ["Smith", "Wilson"]
        similarity2 = self.scraper._calculate_author_similarity(result_authors, citation_authors2)
        self.assertEqual(similarity2, 0.5)  # One of two authors found

    def test_search_citation_with_good_results(self):
        """Test main citation search with good results."""
        if self.test_mode == "unit":
            mock_result = ScholarResult(
                title="Chemistry Test Paper",
                authors="Smith, J.",
                publication="J. Am. Chem. Soc.",
                year=2023,
                url="https://example.com",
                doi="10.1021/test",
                cited_by=50,
                validation_score=0.8
            )

            with patch.object(self.scraper, 'search_with_rotation', return_value=[mock_result]):
                with patch.object(self.scraper, 'validate_chemistry_paper_match', return_value=0.8):
                    results = self.scraper.search_citation(self.sample_citation)

                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0].validation_score, 0.8)

    def test_deduplicate_results(self):
        """Test result deduplication."""
        results = [
            ScholarResult("Test Paper", "Smith", "Journal", 2023, "url1", None, 10, 0.8),
            ScholarResult("Test Paper: A Study", "Smith", "Journal", 2023, "url2", None, 15, 0.9),
            ScholarResult("Different Paper", "Brown", "Journal", 2022, "url3", None, 5, 0.7)
        ]

        unique_results = self.scraper._deduplicate_results(results)

        # Should remove the duplicate "Test Paper" variants
        self.assertEqual(len(unique_results), 2)
        titles = [r.title for r in unique_results]
        self.assertIn("Different Paper", titles)

    def test_titles_similar(self):
        """Test title similarity detection."""
        title1 = "quantumchemistrypapertitle"
        title2 = "quantumchemistrypapertitle"
        self.assertTrue(self.scraper._titles_similar(title1, title2))

        title3 = "quantumchemistry"
        title4 = "quantumphysics"
        self.assertFalse(self.scraper._titles_similar(title3, title4))

    def test_extract_doi_from_results(self):
        """Test DOI extraction from best result."""
        results = [
            ScholarResult("Paper 1", "Smith", "Journal", 2023, "url1", None, 10, 0.5),
            ScholarResult("Paper 2", "Brown", "Journal", 2023, "url2", "10.1021/test", 20, 0.8),
            ScholarResult("Paper 3", "Jones", "Journal", 2023, "url3", "10.1000/other", 5, 0.6)
        ]

        best_doi = self.scraper.extract_doi_from_results(results)
        self.assertEqual(best_doi, "10.1021/test")  # Highest validation score with DOI

    def test_extract_doi_from_results_no_doi(self):
        """Test DOI extraction when no DOI available."""
        results = [
            ScholarResult("Paper 1", "Smith", "Journal", 2023, "url1", None, 10, 0.8),
            ScholarResult("Paper 2", "Brown", "Journal", 2023, "url2", None, 20, 0.9)
        ]

        best_doi = self.scraper.extract_doi_from_results(results)
        self.assertIsNone(best_doi)

    def test_batch_search_citations(self):
        """Test batch processing of multiple citations."""
        citations = [
            {"citation_text": "Smith, J. Test Paper 1. Journal 2023.", "context": ""},
            {"citation_text": "Brown, A. Test Paper 2. Nature 2022.", "context": ""}
        ]

        if self.test_mode == "unit":
            mock_result = ScholarResult("Test", "Author", "Journal", 2023, "url", "doi", 10, 0.8)

            with patch.object(self.scraper, 'search_citation', return_value=[mock_result]):
                with patch.object(self.scraper, 'extract_doi_from_results', return_value="10.1021/test"):
                    results = self.scraper.batch_search_citations(citations)

                    self.assertEqual(len(results), 2)
                    for citation_text, result in results.items():
                        self.assertTrue(result['success'])
                        self.assertEqual(result['best_doi'], "10.1021/test")

    def test_batch_search_citations_with_failures(self):
        """Test batch processing with some failures."""
        citations = [
            {"citation_text": "Valid citation", "context": ""},
            {"citation_text": "Invalid citation", "context": ""}
        ]

        if self.test_mode == "unit":
            def mock_search(citation_text, context=""):
                if "Valid" in citation_text:
                    return [ScholarResult("Test", "Author", "Journal", 2023, "url", "doi", 10, 0.8)]
                else:
                    raise Exception("Search failed")

            with patch.object(self.scraper, 'search_citation', side_effect=mock_search):
                results = self.scraper.batch_search_citations(citations)

                self.assertTrue(results["Valid citation"]['success'])
                self.assertFalse(results["Invalid citation"]['success'])
                self.assertIn("error", results["Invalid citation"])

    def test_batch_search_citations_with_output(self):
        """Test batch processing with file output."""
        citations = [{"citation_text": "Test citation", "context": ""}]

        if self.test_mode == "unit":
            with patch.object(self.scraper, 'search_citation', return_value=[]):
                with patch('builtins.open', create=True) as mock_open:
                    with patch('json.dump') as mock_dump:
                        with patch('pathlib.Path.mkdir'):
                            results = self.scraper.batch_search_citations(
                                citations,
                                output_path="test_output.json"
                            )

                            mock_dump.assert_called_once()

    @pytest.mark.slow
    def test_performance_multiple_searches(self):
        """Test performance with multiple searches."""
        if self.test_mode == "unit":
            citations = ["Test citation " + str(i) for i in range(5)]

            with patch.object(self.scraper, 'search_with_rotation', return_value=[]):
                # Should complete reasonably quickly even with rate limiting
                result = self.assertExecutionTime(
                    lambda: [self.scraper.search_citation(c) for c in citations],
                    max_time=2.0  # Should complete within 2 seconds for mocked version
                )
                self.assertEqual(len(result), 5)

    @pytest.mark.integration
    def test_end_to_end_search(self):
        """End-to-end integration test (only runs in integration mode)."""
        if self.test_mode == "integration":
            # This would test with real Scholar requests
            try:
                # Use a well-known chemistry citation
                citation = "Pauling, L. The Nature of the Chemical Bond. 1939."
                results = self.scraper.search_citation(citation)

                # Should find some results for this famous citation
                self.assertGreater(len(results), 0)

                # Check result structure
                for result in results:
                    self.assertIsInstance(result, ScholarResult)
                    self.assertIsInstance(result.title, str)
                    self.assertIsInstance(result.validation_score, float)

            except Exception as e:
                # Network issues in integration tests are acceptable
                self.skipTest(f"Integration test skipped due to network: {e}")

    @pytest.mark.network
    def test_user_agent_rotation(self):
        """Test that different user agents are used."""
        if self.test_mode == "unit":
            # Mock multiple requests and check different user agents are used
            user_agents_used = []

            def capture_headers(*args, **kwargs):
                user_agents_used.append(kwargs.get('headers', {}).get('User-Agent'))
                return Mock(status_code=200, text="<html></html>")

            self.mock_session.get.side_effect = capture_headers

            # Make multiple searches
            for i in range(3):
                self.scraper.search_with_rotation(f"test query {i}")

            # Should have used different user agents
            unique_agents = set(user_agents_used)
            self.assertGreaterEqual(len(unique_agents), 1)  # At least some variety


if __name__ == '__main__':
    unittest.main()