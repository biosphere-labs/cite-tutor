"""
Tests for Citation extractor module.

Tests both unit and integration modes for citation extraction functionality.
Uses StandardTestCase for basic functionality testing.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, mock_open
import json
import tempfile
from pathlib import Path

from base_test import StandardTestCase
from citation_extractor import ChemistryCitationExtractor, ChemistryCitation


class TestChemistryCitation(unittest.TestCase):
    """Test ChemistryCitation dataclass."""

    def test_chemistry_citation_creation(self):
        """Test creating ChemistryCitation instance."""
        citation = ChemistryCitation(
            citation_text="J. Am. Chem. Soc. 85, 2544 (1963)",
            citation_type="journal_standard",
            context="This paper describes benzene synthesis",
            chemistry_context="synthesis",
            importance_score=0.85,
            compounds_mentioned=["benzene", "C6H6"],
            foundational_relevance="organic_reactions",
            retrieval_priority="high",
            page_location=10,
            section_title="Introduction",
            year=1963,
            authors=["Smith, J."],
            journal="J. Am. Chem. Soc.",
            volume="85",
            pages="2544"
        )

        self.assertEqual(citation.citation_text, "J. Am. Chem. Soc. 85, 2544 (1963)")
        self.assertEqual(citation.citation_type, "journal_standard")
        self.assertEqual(citation.year, 1963)
        self.assertEqual(len(citation.compounds_mentioned), 2)
        self.assertIn("benzene", citation.compounds_mentioned)


class TestChemistryCitationExtractor(StandardTestCase):
    """Test ChemistryCitationExtractor class."""

    def setUp(self):
        super().setUp()
        self.extractor = ChemistryCitationExtractor()

        # Sample chemistry text with various citation formats
        self.sample_text = """
        The concept of resonance was first introduced by Pauling (J. Am. Chem. Soc. 53, 1367 (1931))
        and later developed in his comprehensive work (Pauling, L. The Nature of the Chemical Bond;
        Cornell University Press: Ithaca, 1939). The Diels-Alder reaction was discovered by
        Diels and Alder (Ber. Dtsch. Chem. Ges. 62, 554 (1929)) and has since become one of the
        most important synthetic methods for preparing benzene derivatives and other organic compounds.

        Recent studies have shown (15) that the mechanism involves a concerted process involving
        cyclohexene intermediates. This was confirmed by Brown et al. (1962) in their detailed
        kinetic analysis of the reaction pathway.

        The thermodynamics of this process (Nature 171, 737 (1953)) indicate favorable enthalpy
        changes for the formation of C6H6 derivatives.
        """

        self.sample_section_titles = {
            0: "Introduction",
            5: "Methodology",
            10: "Results",
            15: "Discussion"
        }

    def test_compile_citation_patterns(self):
        """Test compilation of citation regex patterns."""
        patterns = self.extractor._compile_citation_patterns()

        # Check that all expected patterns are compiled
        expected_patterns = [
            'journal_standard', 'journal_alt', 'book', 'author_year',
            'numbered', 'german_journal', 'nature_science'
        ]

        for pattern_name in expected_patterns:
            self.assertIn(pattern_name, patterns)
            self.assertIsNotNone(patterns[pattern_name].pattern)

    def test_compile_compound_patterns(self):
        """Test compilation of compound identification patterns."""
        patterns = self.extractor._compile_compound_patterns()

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

        # Test that patterns can match common compounds
        test_text = "benzene C6H6 methane ethanol"
        found_compounds = set()

        for pattern in patterns:
            matches = pattern.findall(test_text)
            found_compounds.update(matches)

        self.assertIn("benzene", found_compounds)
        self.assertIn("C6H6", found_compounds)

    def test_extract_citations_from_text(self):
        """Test complete citation extraction from text."""
        citations = self.extractor.extract_citations_from_text(self.sample_text, self.sample_section_titles)

        # Should find multiple citations
        self.assertGreater(len(citations), 0)

        # Check that different citation types are found
        citation_types = {c.citation_type for c in citations}
        self.assertIn('journal_standard', citation_types)
        self.assertIn('book', citation_types)

        # Check that citations have required fields
        for citation in citations:
            self.assertIsInstance(citation.citation_text, str)
            self.assertIsInstance(citation.importance_score, float)
            self.assertGreaterEqual(citation.importance_score, 0.0)
            self.assertLessEqual(citation.importance_score, 1.0)

    def test_parse_citation_details_journal_standard(self):
        """Test parsing journal citation details."""
        import re
        pattern = self.extractor.citation_patterns['journal_standard']
        match = pattern.search("J. Am. Chem. Soc. 85, 2544 (1963)")

        self.assertIsNotNone(match)
        details = self.extractor._parse_citation_details(match, 'journal_standard')

        self.assertEqual(details['journal'], "J. Am. Chem. Soc.")
        self.assertEqual(details['volume'], "85")
        self.assertEqual(details['pages'], "2544")
        self.assertEqual(details['year'], 1963)

    def test_parse_citation_details_book(self):
        """Test parsing book citation details."""
        import re
        pattern = self.extractor.citation_patterns['book']
        match = pattern.search("Pauling, L. The Nature of the Chemical Bond; Cornell University Press: Ithaca, 1939")

        self.assertIsNotNone(match)
        details = self.extractor._parse_citation_details(match, 'book')

        self.assertEqual(details['authors'], ["Pauling, L."])
        self.assertEqual(details['year'], 1939)

    def test_parse_citation_details_author_year(self):
        """Test parsing author-year citation details."""
        import re
        pattern = self.extractor.citation_patterns['author_year']
        match = pattern.search("(Brown et al., 1962)")

        self.assertIsNotNone(match)
        details = self.extractor._parse_citation_details(match, 'author_year')

        self.assertEqual(details['authors'], ["Brown et al."])
        self.assertEqual(details['year'], 1962)

    def test_parse_citation_details_german_journal(self):
        """Test parsing German journal citation details."""
        import re
        pattern = self.extractor.citation_patterns['german_journal']
        match = pattern.search("Ber. Dtsch. Chem. Ges. 62, 554 (1929)")

        self.assertIsNotNone(match)
        details = self.extractor._parse_citation_details(match, 'german_journal')

        self.assertEqual(details['journal'], "Ber. Dtsch. Chem. Ges.")
        self.assertEqual(details['volume'], "62")
        self.assertEqual(details['pages'], "554")
        self.assertEqual(details['year'], 1929)

    def test_parse_citation_details_nature_science(self):
        """Test parsing Nature/Science citation details."""
        import re
        pattern = self.extractor.citation_patterns['nature_science']
        match = pattern.search("Nature 171, 737 (1953)")

        self.assertIsNotNone(match)
        details = self.extractor._parse_citation_details(match, 'nature_science')

        self.assertEqual(details['journal'], "Nature")
        self.assertEqual(details['volume'], "171")
        self.assertEqual(details['pages'], "737")
        self.assertEqual(details['year'], 1953)

    def test_extract_citation_context(self):
        """Test extraction of citation context."""
        # Test context extraction around a known citation
        test_text = "Previous work showed interesting results. The concept was introduced (J. Am. Chem. Soc. 53, 1367 (1931)) by Pauling. This was revolutionary for chemistry."

        # Find the citation position
        citation_start = test_text.find("(J. Am. Chem. Soc.")
        citation_end = citation_start + len("(J. Am. Chem. Soc. 53, 1367 (1931))")

        context = self.extractor._extract_citation_context(test_text, citation_start, citation_end, 0)

        # Should include surrounding sentences
        self.assertIn("concept was introduced", context)
        self.assertIn("revolutionary for chemistry", context)

    def test_classify_chemistry_context(self):
        """Test chemistry context classification."""
        # Test synthesis context
        synthesis_context = "The synthesis of benzene was achieved using a new catalytic method with high yield."
        context_type = self.extractor._classify_chemistry_context(synthesis_context)
        self.assertEqual(context_type, 'synthesis')

        # Test mechanism context
        mechanism_context = "The reaction mechanism involves an intermediate transition state."
        context_type = self.extractor._classify_chemistry_context(mechanism_context)
        self.assertEqual(context_type, 'mechanism')

        # Test theory context
        theory_context = "The molecular orbital theory explains the bonding in this compound."
        context_type = self.extractor._classify_chemistry_context(theory_context)
        self.assertEqual(context_type, 'theory')

        # Test structure context
        structure_context = "The crystal structure shows interesting geometry and bond angles."
        context_type = self.extractor._classify_chemistry_context(structure_context)
        self.assertEqual(context_type, 'structure')

        # Test general context
        general_context = "This is a general statement about chemistry."
        context_type = self.extractor._classify_chemistry_context(general_context)
        self.assertEqual(context_type, 'general')

    def test_extract_compounds_from_context(self):
        """Test compound extraction from context."""
        context = "The synthesis of benzene (C6H6) and toluene from methane and ethanol was successful."

        compounds = self.extractor._extract_compounds_from_context(context)

        self.assertIn("benzene", compounds)
        self.assertIn("C6H6", compounds)
        self.assertIn("toluene", compounds)
        self.assertIn("methane", compounds)
        self.assertIn("ethanol", compounds)

        # Should filter out common false positives
        self.assertNotIn("The", compounds)
        self.assertNotIn("And", compounds)

    def test_get_section_title(self):
        """Test section title retrieval."""
        section_titles = {0: "Introduction", 5: "Methods", 10: "Results"}

        # Test exact match
        title = self.extractor._get_section_title(5, section_titles)
        self.assertEqual(title, "Methods")

        # Test within section
        title = self.extractor._get_section_title(7, section_titles)
        self.assertEqual(title, "Methods")

        # Test before first section
        title = self.extractor._get_section_title(2, section_titles)
        self.assertEqual(title, "Introduction")

        # Test no section titles
        title = self.extractor._get_section_title(5, None)
        self.assertEqual(title, "Unknown Section")

    def test_calculate_citation_priority_historical(self):
        """Test priority calculation for historical citations."""
        # Old citation should get high priority
        old_citation = {'year': 1925, 'journal': 'J. Am. Chem. Soc.', 'authors': ['Pauling']}
        context = "This foundational work on chemical bonding theory was revolutionary."
        chemistry_context = "theory"

        score = self.extractor._calculate_citation_priority(old_citation, context, chemistry_context)

        # Should get high score for old year, important journal, foundational author, and theory context
        self.assertGreater(score, 0.5)

    def test_calculate_citation_priority_modern(self):
        """Test priority calculation for modern citations."""
        # Modern citation should get lower priority
        modern_citation = {'year': 2020, 'journal': 'Some Journal', 'authors': ['Smith']}
        context = "Recent work shows interesting results."
        chemistry_context = "general"

        score = self.extractor._calculate_citation_priority(modern_citation, context, chemistry_context)

        # Should get lower score for recent year and general context
        self.assertLess(score, 0.3)

    def test_get_journal_impact_score(self):
        """Test journal impact scoring."""
        # Test high impact journals
        self.assertEqual(self.extractor._get_journal_impact_score("Nature"), 1.0)
        self.assertEqual(self.extractor._get_journal_impact_score("J. Am. Chem. Soc."), 1.0)

        # Test medium impact journals
        self.assertEqual(self.extractor._get_journal_impact_score("Angew. Chem."), 0.7)

        # Test historical journals
        self.assertEqual(self.extractor._get_journal_impact_score("Ber. Dtsch. Chem. Ges."), 0.8)

        # Test general chemistry journals
        self.assertEqual(self.extractor._get_journal_impact_score("Journal of Chemistry"), 0.5)

        # Test non-chemistry journals
        self.assertEqual(self.extractor._get_journal_impact_score("Physics Review"), 0.3)

    def test_assess_foundational_relevance(self):
        """Test foundational relevance assessment."""
        # Test valence theory
        citation_details = {'journal': '', 'authors': ['Lewis']}
        context = "The concept of electron pair bonding and valence theory was developed."
        relevance = self.extractor._assess_foundational_relevance(citation_details, context)
        self.assertEqual(relevance, 'valence_theory')

        # Test orbital theory
        citation_details = {'journal': '', 'authors': []}
        context = "Molecular orbital calculations and quantum mechanical treatment."
        relevance = self.extractor._assess_foundational_relevance(citation_details, context)
        self.assertEqual(relevance, 'orbital_theory')

        # Test organic reactions
        citation_details = {'journal': '', 'authors': ['Grignard']}
        context = "The Grignard reaction mechanism and synthetic applications."
        relevance = self.extractor._assess_foundational_relevance(citation_details, context)
        self.assertEqual(relevance, 'organic_reactions')

        # Test general chemistry
        citation_details = {'journal': '', 'authors': []}
        context = "Some general chemistry information."
        relevance = self.extractor._assess_foundational_relevance(citation_details, context)
        self.assertEqual(relevance, 'general_chemistry')

    def test_determine_retrieval_priority(self):
        """Test retrieval priority determination."""
        self.assertEqual(self.extractor._determine_retrieval_priority(0.9), 'critical')
        self.assertEqual(self.extractor._determine_retrieval_priority(0.7), 'high')
        self.assertEqual(self.extractor._determine_retrieval_priority(0.5), 'medium')
        self.assertEqual(self.extractor._determine_retrieval_priority(0.3), 'low')
        self.assertEqual(self.extractor._determine_retrieval_priority(0.1), 'minimal')

    def test_classify_citation_type(self):
        """Test citation type classification."""
        # Test journal citation
        journal_citation = "J. Am. Chem. Soc. 85, 2544 (1963)"
        citation_type = self.extractor.classify_citation_type(journal_citation)
        self.assertEqual(citation_type, 'journal_standard')

        # Test book citation
        book_citation = "Pauling, L. The Nature of the Chemical Bond; Cornell University Press: Ithaca, 1939"
        citation_type = self.extractor.classify_citation_type(book_citation)
        self.assertEqual(citation_type, 'book')

        # Test author-year citation
        author_year_citation = "(Brown et al., 1962)"
        citation_type = self.extractor.classify_citation_type(author_year_citation)
        self.assertEqual(citation_type, 'author_year')

        # Test numbered citation
        numbered_citation = "(15)"
        citation_type = self.extractor.classify_citation_type(numbered_citation)
        self.assertEqual(citation_type, 'numbered')

        # Test unknown citation
        unknown_citation = "Some random text"
        citation_type = self.extractor.classify_citation_type(unknown_citation)
        self.assertEqual(citation_type, 'unknown')

    def test_prioritize_citations_for_lookup(self):
        """Test citation prioritization for lookup."""
        citations = [
            ChemistryCitation(
                citation_text="Citation 1", citation_type="journal", context="", chemistry_context="",
                importance_score=0.5, compounds_mentioned=[], foundational_relevance="",
                retrieval_priority="low", page_location=1, section_title=""
            ),
            ChemistryCitation(
                citation_text="Citation 2", citation_type="journal", context="", chemistry_context="",
                importance_score=0.9, compounds_mentioned=[], foundational_relevance="",
                retrieval_priority="critical", page_location=2, section_title=""
            ),
            ChemistryCitation(
                citation_text="Citation 3", citation_type="journal", context="", chemistry_context="",
                importance_score=0.7, compounds_mentioned=[], foundational_relevance="",
                retrieval_priority="high", page_location=3, section_title=""
            )
        ]

        prioritized = self.extractor.prioritize_citations_for_lookup(citations)

        # Should be sorted by priority then importance score
        self.assertEqual(prioritized[0].retrieval_priority, "critical")
        self.assertEqual(prioritized[1].retrieval_priority, "high")
        self.assertEqual(prioritized[2].retrieval_priority, "low")

    def test_identify_foundational_papers(self):
        """Test identification of foundational papers."""
        citations = [
            ChemistryCitation(
                citation_text="Old important paper", citation_type="journal", context="", chemistry_context="",
                importance_score=0.8, compounds_mentioned=[], foundational_relevance="",
                retrieval_priority="critical", page_location=1, section_title="", year=1925
            ),
            ChemistryCitation(
                citation_text="Modern paper", citation_type="journal", context="", chemistry_context="",
                importance_score=0.8, compounds_mentioned=[], foundational_relevance="",
                retrieval_priority="high", page_location=2, section_title="", year=1985
            ),
            ChemistryCitation(
                citation_text="Low importance old paper", citation_type="journal", context="", chemistry_context="",
                importance_score=0.3, compounds_mentioned=[], foundational_relevance="",
                retrieval_priority="low", page_location=3, section_title="", year=1920
            )
        ]

        foundational = self.extractor.identify_foundational_papers(citations)

        # Should only include old, high-importance papers
        self.assertEqual(len(foundational), 1)
        self.assertEqual(foundational[0].year, 1925)

    def test_export_citations_to_json(self):
        """Test citation export to JSON."""
        citations = [
            ChemistryCitation(
                citation_text="Test citation", citation_type="journal", context="test context",
                chemistry_context="synthesis", importance_score=0.7, compounds_mentioned=["benzene"],
                foundational_relevance="organic", retrieval_priority="high", page_location=1,
                section_title="Introduction", year=1950, authors=["Smith"], journal="Test Journal",
                volume="1", pages="123"
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            self.extractor.export_citations_to_json(citations, temp_path)

            # Check that file was created and contains correct data
            self.assertTrue(Path(temp_path).exists())

            with open(temp_path, 'r') as f:
                data = json.load(f)

            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['citation_text'], "Test citation")
            self.assertEqual(data[0]['year'], 1950)
            self.assertEqual(data[0]['compounds_mentioned'], ["benzene"])

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_citation_summary(self):
        """Test citation summary generation."""
        citations = [
            ChemistryCitation(
                citation_text="Citation 1", citation_type="journal_standard", context="",
                chemistry_context="synthesis", importance_score=0.8, compounds_mentioned=["benzene", "toluene"],
                foundational_relevance="organic", retrieval_priority="high", page_location=1,
                section_title="", year=1930
            ),
            ChemistryCitation(
                citation_text="Citation 2", citation_type="book", context="",
                chemistry_context="theory", importance_score=0.9, compounds_mentioned=["C6H6"],
                foundational_relevance="valence", retrieval_priority="critical", page_location=2,
                section_title="", year=1935
            )
        ]

        summary = self.extractor.generate_citation_summary(citations)

        # Check summary structure
        self.assertEqual(summary['total_citations'], 2)
        self.assertIn('average_importance_score', summary)
        self.assertIn('type_distribution', summary)
        self.assertIn('priority_distribution', summary)
        self.assertIn('context_distribution', summary)
        self.assertIn('decade_distribution', summary)
        self.assertIn('top_compounds', summary)

        # Check specific values
        self.assertEqual(summary['type_distribution']['journal_standard'], 1)
        self.assertEqual(summary['type_distribution']['book'], 1)
        self.assertEqual(summary['priority_distribution']['high'], 1)
        self.assertEqual(summary['priority_distribution']['critical'], 1)
        self.assertEqual(summary['decade_distribution']['1930s'], 2)

    def test_generate_citation_summary_empty(self):
        """Test citation summary with empty list."""
        summary = self.extractor.generate_citation_summary([])
        self.assertEqual(summary, {'total_citations': 0})

    def test_foundational_authors_detection(self):
        """Test detection of foundational chemistry authors."""
        # Test citation with foundational author
        citation_details = {'journal': 'J. Am. Chem. Soc.', 'authors': ['Pauling'], 'year': 1931}
        context = "Chemical bonding theory"
        chemistry_context = "theory"

        score = self.extractor._calculate_citation_priority(citation_details, context, chemistry_context)

        # Should get bonus for foundational author
        self.assertGreater(score, 0.5)

        # Test with non-foundational author
        citation_details_2 = {'journal': 'J. Am. Chem. Soc.', 'authors': ['Unknown'], 'year': 1931}
        score_2 = self.extractor._calculate_citation_priority(citation_details_2, context, chemistry_context)

        # Should have lower score without foundational author bonus
        self.assertLess(score_2, score)

    def test_chemistry_journal_recognition(self):
        """Test recognition of chemistry journals."""
        # Test various journal abbreviations
        journal_abbreviations = [
            "j. am. chem. soc.",
            "jacs",
            "angew. chem.",
            "ber. dtsch. chem. ges.",
            "ber.",
            "nature",
            "science"
        ]

        for abbrev in journal_abbreviations:
            self.assertIn(abbrev, self.extractor.CHEMISTRY_JOURNALS)

    def test_compound_pattern_matching(self):
        """Test compound pattern matching."""
        test_compounds = [
            "C6H6",      # Simple formula
            "benzene",   # Common name
            "ethyl",     # Functional group
            "methanol",  # Systematic name
            "H2SO4",     # Inorganic compound
            "CH3COOH"    # Organic acid
        ]

        for compound in test_compounds:
            found = False
            for pattern in self.extractor.compound_patterns:
                if pattern.search(compound):
                    found = True
                    break
            self.assertTrue(found, f"Compound '{compound}' not matched by any pattern")

    @pytest.mark.slow
    def test_performance_large_text(self):
        """Test performance with large text."""
        # Create large text with many citations
        large_text = self.sample_text * 100

        # Test extraction performance
        result = self.assertExecutionTime(
            self.extractor.extract_citations_from_text,
            max_time=10.0,  # Should complete within 10 seconds
            text=large_text
        )

        self.assertIsInstance(result, list)

    def test_edge_cases_malformed_citations(self):
        """Test handling of malformed citations."""
        malformed_text = """
        This text has some malformed citations like (J. Am. Chem. Soc. incomplete
        and (1999) without proper journal info, and Nature
        missing volume and page info.
        """

        # Should not crash on malformed citations
        citations = self.extractor.extract_citations_from_text(malformed_text)
        self.assertIsInstance(citations, list)

    def test_context_length_limitation(self):
        """Test context length limitation."""
        # Create very long context
        long_sentence = "This is a very long sentence. " * 100
        long_text = f"{long_sentence} (J. Am. Chem. Soc. 85, 2544 (1963)) {long_sentence}"

        citations = self.extractor.extract_citations_from_text(long_text)

        # Context should be limited
        for citation in citations:
            self.assertLessEqual(len(citation.context), 503)  # 500 + "..."

    def test_duplicate_citation_handling(self):
        """Test handling of duplicate citations."""
        text_with_duplicates = """
        The first mention (J. Am. Chem. Soc. 85, 2544 (1963)) shows this.
        Again, the same citation (J. Am. Chem. Soc. 85, 2544 (1963)) appears here.
        """

        citations = self.extractor.extract_citations_from_text(text_with_duplicates)

        # Should find both instances (they may have different contexts)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].citation_text, citations[1].citation_text)

    def test_section_title_assignment(self):
        """Test correct section title assignment."""
        section_titles = {0: "Abstract", 3: "Introduction", 8: "Methods"}
        text_lines = [
            "Abstract content",
            "More abstract",
            "End abstract",
            "Introduction starts",
            "Citation here (J. Am. Chem. Soc. 85, 2544 (1963))",
            "More intro",
            "Still intro",
            "End intro",
            "Methods section",
            "Another citation (Nature 171, 737 (1953))"
        ]

        text = "\n".join(text_lines)
        citations = self.extractor.extract_citations_from_text(text, section_titles)

        # Check section assignments
        if len(citations) >= 2:
            # First citation should be in Introduction section
            self.assertEqual(citations[0].section_title, "Introduction")
            # Second citation should be in Methods section
            self.assertEqual(citations[1].section_title, "Methods")


if __name__ == '__main__':
    unittest.main()