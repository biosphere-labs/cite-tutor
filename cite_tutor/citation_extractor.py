"""
Citation extraction module specialized for chemistry literature.
Focuses on pre-1960s foundational papers with chemistry-specific patterns.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChemistryCitation:
    """Data class for chemistry citation information."""
    citation_text: str
    citation_type: str
    context: str
    chemistry_context: str
    importance_score: float
    compounds_mentioned: List[str]
    foundational_relevance: str
    retrieval_priority: str
    page_location: int
    section_title: str
    year: Optional[int] = None
    authors: List[str] = None
    journal: str = ""
    volume: str = ""
    pages: str = ""


class ChemistryCitationExtractor:
    """
    Specialized citation extractor for chemistry literature.
    Focuses on foundational papers and chemistry-specific citation patterns.
    """

    # Chemistry journal abbreviations and full names
    CHEMISTRY_JOURNALS = {
        "j. am. chem. soc.": "Journal of the American Chemical Society",
        "j.am.chem.soc.": "Journal of the American Chemical Society",
        "jacs": "Journal of the American Chemical Society",
        "angew. chem.": "Angewandte Chemie",
        "angew.chem.": "Angewandte Chemie",
        "ber. dtsch. chem. ges.": "Berichte der Deutschen Chemischen Gesellschaft",
        "ber.dtsch.chem.ges.": "Berichte der Deutschen Chemischen Gesellschaft",
        "ber.": "Berichte der Deutschen Chemischen Gesellschaft",
        "j. org. chem.": "Journal of Organic Chemistry",
        "j.org.chem.": "Journal of Organic Chemistry",
        "j. phys. chem.": "Journal of Physical Chemistry",
        "j.phys.chem.": "Journal of Physical Chemistry",
        "chem. rev.": "Chemical Reviews",
        "chem.rev.": "Chemical Reviews",
        "nature": "Nature",
        "science": "Science",
        "proc. natl. acad. sci.": "Proceedings of the National Academy of Sciences",
        "j. chem. phys.": "Journal of Chemical Physics",
        "j.chem.phys.": "Journal of Chemical Physics",
        "tetrahedron": "Tetrahedron",
        "tetrahedron lett.": "Tetrahedron Letters",
        "chem. ber.": "Chemische Berichte",
        "ann. chem.": "Annalen der Chemie",
        "liebigs ann.": "Liebigs Annalen der Chemie",
        "j. chem. soc.": "Journal of the Chemical Society",
        "j.chem.soc.": "Journal of the Chemical Society",
        "chem. commun.": "Chemical Communications",
        "acc. chem. res.": "Accounts of Chemical Research"
    }

    # Foundational chemistry authors
    FOUNDATIONAL_AUTHORS = [
        "pauling", "woodward", "corey", "hückel", "huckel", "mulliken",
        "robinson", "ingold", "winstein", "brown", "bartlett", "fischer",
        "kekulé", "kekule", "vant hoff", "le bel", "arrhenius", "ostwald",
        "nernst", "lewis", "langmuir", "sidgwick", "fajans", "born",
        "haber", "bosch", "grignard", "sabatier", "werner", "coordination",
        "diels", "alder", "wittig", "cope", "claisen", "aldol", "mannich",
        "friedel", "crafts", "wurtz", "kolbe", "perkin", "hofmann"
    ]

    # Chemistry context keywords
    CHEMISTRY_CONTEXTS = {
        'synthesis': [
            'synthesis', 'synthesize', 'preparation', 'prepared', 'reaction',
            'yield', 'product', 'reagent', 'catalyst', 'procedure'
        ],
        'mechanism': [
            'mechanism', 'pathway', 'intermediate', 'transition', 'state',
            'catalysis', 'kinetics', 'rate', 'elementary', 'step'
        ],
        'structure': [
            'structure', 'conformation', 'configuration', 'stereochemistry',
            'geometry', 'bond', 'angle', 'length', 'crystal'
        ],
        'theory': [
            'theory', 'theoretical', 'quantum', 'molecular', 'orbital',
            'electronic', 'valence', 'hybridization', 'resonance'
        ],
        'spectroscopy': [
            'spectroscopy', 'nmr', 'infrared', 'ultraviolet', 'mass',
            'spectrometry', 'x-ray', 'diffraction', 'analysis'
        ],
        'thermodynamics': [
            'thermodynamics', 'enthalpy', 'entropy', 'free energy', 'equilibrium',
            'constant', 'temperature', 'pressure', 'phase'
        ]
    }

    def __init__(self):
        self.citation_patterns = self._compile_citation_patterns()
        self.compound_patterns = self._compile_compound_patterns()

    def _compile_citation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for different citation types."""

        patterns = {
            # Journal citations: "J. Am. Chem. Soc. 85, 2544 (1963)"
            'journal_standard': re.compile(
                r'([A-Z][a-z]*\.?\s*(?:[A-Z][a-z]*\.?\s*)*)\s*'  # Journal name
                r'(\d{1,4}),?\s*'  # Volume
                r'(\d+)\s*'  # Page
                r'\((\d{4})\)',  # Year
                re.IGNORECASE
            ),

            # Alternative journal format: "J. Am. Chem. Soc., 1963, 85, 2544"
            'journal_alt': re.compile(
                r'([A-Z][a-z]*\.?\s*(?:[A-Z][a-z]*\.?\s*)*),?\s*'  # Journal
                r'(\d{4}),?\s*'  # Year
                r'(\d{1,4}),?\s*'  # Volume
                r'(\d+)',  # Page
                re.IGNORECASE
            ),

            # Book citations: "Pauling, L. The Nature of the Chemical Bond; Cornell University Press: Ithaca, 1960"
            'book': re.compile(
                r'([A-Z][a-z]+(?:,\s*[A-Z]\.?)*)\s+'  # Author
                r'([^;]+);?\s*'  # Title
                r'([^:]+):\s*'  # Publisher
                r'([^,]+),?\s*'  # Location
                r'(\d{4})',  # Year
                re.IGNORECASE
            ),

            # Author-year format: "(Brown et al., 1962)"
            'author_year': re.compile(
                r'\(([A-Z][a-z]+(?:\s+et al\.?)?),?\s*(\d{4})\)',
                re.IGNORECASE
            ),

            # Numbered references: "...as shown previously (15)"
            'numbered': re.compile(
                r'\((\d{1,3})\)'
            ),

            # German/European journals: "Ber. Dtsch. Chem. Ges. 45, 1123 (1912)"
            'german_journal': re.compile(
                r'(Ber\.?\s*Dtsch\.?\s*Chem\.?\s*Ges\.?|'
                r'Ann\.?\s*Chem\.?|'
                r'Liebigs\s*Ann\.?)\s*'  # German journal names
                r'(\d{1,4}),?\s*'  # Volume
                r'(\d+)\s*'  # Page
                r'\((\d{4})\)',  # Year
                re.IGNORECASE
            ),

            # Nature/Science format: "Nature 171, 737 (1953)"
            'nature_science': re.compile(
                r'(Nature|Science)\s+'
                r'(\d{1,4}),?\s*'  # Volume
                r'(\d+)\s*'  # Page
                r'\((\d{4})\)',  # Year
                re.IGNORECASE
            )
        }

        return patterns

    def _compile_compound_patterns(self) -> List[re.Pattern]:
        """Compile patterns to identify chemical compounds in text."""

        return [
            # Basic chemical formulas
            re.compile(r'\b[A-Z][a-z]?(?:\d+[A-Z][a-z]?\d*)*\b'),

            # Common organic compounds
            re.compile(r'\b(?:methane|ethane|propane|butane|benzene|toluene|phenol|'
                      r'acetone|ethanol|methanol|acetic acid|formic acid)\b', re.IGNORECASE),

            # Systematic names
            re.compile(r'\b\d+[,-]\w+(?:[,-]\d+[,-]\w+)*\b'),

            # Complex molecular names
            re.compile(r'\b[a-z]+(?:yl|ene|ane|yne|ol|al|one|oic acid|amide)\b', re.IGNORECASE)
        ]

    def extract_citations_from_text(self, text: str, section_titles: Optional[Dict[int, str]] = None) -> List[ChemistryCitation]:
        """
        Extract all chemistry citations from text with context analysis.
        """

        citations = []

        # Split text into lines for position tracking
        lines = text.split('\n')

        for line_num, line in enumerate(lines):
            # Try each citation pattern
            for pattern_name, pattern in self.citation_patterns.items():
                matches = pattern.finditer(line)

                for match in matches:
                    # Extract basic citation info
                    citation_text = match.group(0)

                    # Parse citation details based on pattern type
                    citation_details = self._parse_citation_details(match, pattern_name)

                    # Extract context around citation
                    context = self._extract_citation_context(text, match.start(), match.end(), line_num)

                    # Determine chemistry context
                    chemistry_context = self._classify_chemistry_context(context)

                    # Extract compounds mentioned in context
                    compounds = self._extract_compounds_from_context(context)

                    # Determine section title
                    section_title = self._get_section_title(line_num, section_titles)

                    # Calculate importance score
                    importance_score = self._calculate_citation_priority(citation_details, context, chemistry_context)

                    # Assess foundational relevance
                    foundational_relevance = self._assess_foundational_relevance(citation_details, context)

                    # Determine retrieval priority
                    retrieval_priority = self._determine_retrieval_priority(importance_score)

                    # Create citation object
                    citation = ChemistryCitation(
                        citation_text=citation_text,
                        citation_type=pattern_name,
                        context=context,
                        chemistry_context=chemistry_context,
                        importance_score=importance_score,
                        compounds_mentioned=compounds,
                        foundational_relevance=foundational_relevance,
                        retrieval_priority=retrieval_priority,
                        page_location=line_num,
                        section_title=section_title,
                        **citation_details
                    )

                    citations.append(citation)

        logger.info(f"Extracted {len(citations)} citations from text")
        return citations

    def _parse_citation_details(self, match: re.Match, pattern_name: str) -> Dict:
        """Parse citation details based on pattern type."""

        details = {'year': None, 'authors': [], 'journal': '', 'volume': '', 'pages': ''}

        if pattern_name == 'journal_standard':
            details.update({
                'journal': match.group(1).strip(),
                'volume': match.group(2),
                'pages': match.group(3),
                'year': int(match.group(4))
            })

        elif pattern_name == 'journal_alt':
            details.update({
                'journal': match.group(1).strip(),
                'year': int(match.group(2)),
                'volume': match.group(3),
                'pages': match.group(4)
            })

        elif pattern_name == 'book':
            authors = [match.group(1).strip()]
            details.update({
                'authors': authors,
                'year': int(match.group(5))
            })

        elif pattern_name == 'author_year':
            authors = [match.group(1).strip()]
            details.update({
                'authors': authors,
                'year': int(match.group(2))
            })

        elif pattern_name in ['german_journal', 'nature_science']:
            details.update({
                'journal': match.group(1).strip(),
                'volume': match.group(2),
                'pages': match.group(3),
                'year': int(match.group(4))
            })

        return details

    def _extract_citation_context(self, text: str, start_pos: int, end_pos: int, line_num: int) -> str:
        """Extract context around citation for analysis."""

        # Get surrounding sentences
        sentences = re.split(r'[.!?]+', text)

        # Find sentence containing citation
        current_pos = 0
        context_sentences = []

        for i, sentence in enumerate(sentences):
            sentence_start = current_pos
            sentence_end = current_pos + len(sentence)

            # If citation is in this sentence, include it and surrounding sentences
            if sentence_start <= start_pos <= sentence_end:
                # Include previous sentence if available
                if i > 0:
                    context_sentences.append(sentences[i-1])

                # Include current sentence
                context_sentences.append(sentence)

                # Include next sentence if available
                if i < len(sentences) - 1:
                    context_sentences.append(sentences[i+1])

                break

            current_pos = sentence_end + 1

        context = '. '.join(context_sentences).strip()

        # Limit context length
        if len(context) > 500:
            context = context[:500] + "..."

        return context

    def _classify_chemistry_context(self, context: str) -> str:
        """Classify the chemistry context of a citation."""

        context_lower = context.lower()
        scores = {}

        for context_type, keywords in self.CHEMISTRY_CONTEXTS.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                scores[context_type] = score

        if scores:
            return max(scores, key=scores.get)
        else:
            return 'general'

    def _extract_compounds_from_context(self, context: str) -> List[str]:
        """Extract chemical compounds mentioned in citation context."""

        compounds = set()

        for pattern in self.compound_patterns:
            matches = pattern.findall(context)
            compounds.update(matches)

        # Filter out common false positives
        false_positives = {'The', 'And', 'For', 'This', 'All', 'One', 'Two', 'New', 'Old'}
        compounds = compounds - false_positives

        # Limit to top 10 most relevant compounds
        return list(compounds)[:10]

    def _get_section_title(self, line_num: int, section_titles: Optional[Dict[int, str]]) -> str:
        """Get section title for citation location."""

        if not section_titles:
            return "Unknown Section"

        # Find the most recent section title
        for section_line in sorted(section_titles.keys(), reverse=True):
            if section_line <= line_num:
                return section_titles[section_line]

        return "Introduction"

    def _calculate_citation_priority(self, citation_details: Dict, context: str, chemistry_context: str) -> float:
        """Calculate priority score for chemistry citations."""

        score = 0.0

        # Time period bonus (older = more foundational)
        year = citation_details.get('year')
        if year:
            if year < 1900: score += 3.0
            elif year < 1920: score += 2.5
            elif year < 1940: score += 2.0
            elif year < 1960: score += 1.5
            elif year < 1980: score += 1.0

        # Author importance bonus
        citation_text = (citation_details.get('journal', '') + ' ' +
                        ' '.join(citation_details.get('authors', []))).lower()

        for author in self.FOUNDATIONAL_AUTHORS:
            if author in citation_text:
                score += 1.5
                break

        # Journal importance bonus
        journal_score = self._get_journal_impact_score(citation_details.get('journal', ''))
        score += journal_score

        # Chemistry context importance
        context_scores = {
            'mechanism': 1.2,
            'synthesis': 1.1,
            'theory': 1.3,
            'structure': 1.0,
            'spectroscopy': 0.8,
            'thermodynamics': 0.9,
            'general': 0.5
        }
        score += context_scores.get(chemistry_context, 0.5)

        # Context quality bonus
        if len(context) > 200:  # Substantial context
            score += 0.3

        # Normalize score to 0-1 range
        return min(1.0, score / 10.0)

    def _get_journal_impact_score(self, journal: str) -> float:
        """Get impact score for chemistry journals."""

        journal_lower = journal.lower().strip()

        # High impact journals
        high_impact = ['nature', 'science', 'j. am. chem. soc.', 'jacs']
        if any(hi in journal_lower for hi in high_impact):
            return 1.0

        # Medium impact chemistry journals
        medium_impact = ['angew. chem.', 'chem. rev.', 'j. org. chem.', 'j. phys. chem.']
        if any(mi in journal_lower for mi in medium_impact):
            return 0.7

        # Historical chemistry journals
        historical = ['ber. dtsch. chem. ges.', 'ber.', 'ann. chem.', 'liebigs ann.']
        if any(hist in journal_lower for hist in historical):
            return 0.8

        # Other chemistry journals
        if any(word in journal_lower for word in ['chem', 'chemical']):
            return 0.5

        return 0.3

    def _assess_foundational_relevance(self, citation_details: Dict, context: str) -> str:
        """Assess foundational relevance to chemistry concepts."""

        context_lower = context.lower()
        citation_text = (citation_details.get('journal', '') + ' ' +
                        ' '.join(citation_details.get('authors', []))).lower()

        # Check for specific foundational concepts
        foundational_concepts = {
            'valence_theory': ['valence', 'bond', 'lewis', 'electron pair'],
            'orbital_theory': ['orbital', 'molecular orbital', 'quantum', 'wave function'],
            'stereochemistry': ['stereochemistry', 'configuration', 'conformation', 'chirality'],
            'reaction_mechanisms': ['mechanism', 'intermediate', 'transition state', 'catalysis'],
            'thermodynamics': ['thermodynamics', 'enthalpy', 'entropy', 'equilibrium'],
            'periodic_trends': ['periodic', 'electronegativity', 'atomic radius', 'ionization'],
            'coordination_chemistry': ['coordination', 'complex', 'ligand', 'werner'],
            'organic_reactions': ['grignard', 'diels alder', 'friedel crafts', 'aldol', 'wittig'],
            'crystallography': ['crystal', 'diffraction', 'structure', 'x-ray'],
            'spectroscopy': ['spectroscopy', 'nmr', 'infrared', 'mass spectrometry']
        }

        for concept, keywords in foundational_concepts.items():
            if any(keyword in context_lower or keyword in citation_text for keyword in keywords):
                return concept

        return 'general_chemistry'

    def _determine_retrieval_priority(self, importance_score: float) -> str:
        """Determine retrieval priority based on importance score."""

        if importance_score >= 0.8:
            return 'critical'
        elif importance_score >= 0.6:
            return 'high'
        elif importance_score >= 0.4:
            return 'medium'
        elif importance_score >= 0.2:
            return 'low'
        else:
            return 'minimal'

    def classify_citation_type(self, citation: str) -> str:
        """Classify citation type for a given citation string."""

        for pattern_name, pattern in self.citation_patterns.items():
            if pattern.search(citation):
                return pattern_name

        return 'unknown'

    def prioritize_citations_for_lookup(self, citations: List[ChemistryCitation]) -> List[ChemistryCitation]:
        """Sort citations by retrieval priority and importance."""

        # Define priority order
        priority_order = {'critical': 5, 'high': 4, 'medium': 3, 'low': 2, 'minimal': 1}

        # Sort by priority and then by importance score
        sorted_citations = sorted(
            citations,
            key=lambda c: (priority_order.get(c.retrieval_priority, 0), c.importance_score),
            reverse=True
        )

        logger.info(f"Prioritized {len(sorted_citations)} citations for lookup")
        return sorted_citations

    def identify_foundational_papers(self, citations: List[ChemistryCitation]) -> List[ChemistryCitation]:
        """Identify foundational papers from citation list."""

        foundational = [
            c for c in citations
            if (c.importance_score >= 0.6 and
                c.retrieval_priority in ['critical', 'high'] and
                c.year and c.year < 1970)
        ]

        logger.info(f"Identified {len(foundational)} foundational papers")
        return foundational

    def export_citations_to_json(self, citations: List[ChemistryCitation], output_path: str):
        """Export citations to JSON format for downstream processing."""

        citations_data = []
        for citation in citations:
            citations_data.append({
                'citation_text': citation.citation_text,
                'citation_type': citation.citation_type,
                'context': citation.context,
                'chemistry_context': citation.chemistry_context,
                'importance_score': citation.importance_score,
                'compounds_mentioned': citation.compounds_mentioned,
                'foundational_relevance': citation.foundational_relevance,
                'retrieval_priority': citation.retrieval_priority,
                'page_location': citation.page_location,
                'section_title': citation.section_title,
                'year': citation.year,
                'authors': citation.authors,
                'journal': citation.journal,
                'volume': citation.volume,
                'pages': citation.pages
            })

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(citations_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(citations)} citations to {output_path}")

    def generate_citation_summary(self, citations: List[ChemistryCitation]) -> Dict:
        """Generate summary statistics for extracted citations."""

        if not citations:
            return {'total_citations': 0}

        # Count by type
        type_counts = {}
        for citation in citations:
            type_counts[citation.citation_type] = type_counts.get(citation.citation_type, 0) + 1

        # Count by priority
        priority_counts = {}
        for citation in citations:
            priority_counts[citation.retrieval_priority] = priority_counts.get(citation.retrieval_priority, 0) + 1

        # Count by chemistry context
        context_counts = {}
        for citation in citations:
            context_counts[citation.chemistry_context] = context_counts.get(citation.chemistry_context, 0) + 1

        # Decade distribution
        decade_counts = {}
        for citation in citations:
            if citation.year:
                decade = (citation.year // 10) * 10
                decade_counts[f"{decade}s"] = decade_counts.get(f"{decade}s", 0) + 1

        # Top compounds
        all_compounds = []
        for citation in citations:
            all_compounds.extend(citation.compounds_mentioned)

        compound_counts = {}
        for compound in all_compounds:
            compound_counts[compound] = compound_counts.get(compound, 0) + 1

        top_compounds = sorted(compound_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            'total_citations': len(citations),
            'average_importance_score': sum(c.importance_score for c in citations) / len(citations),
            'type_distribution': type_counts,
            'priority_distribution': priority_counts,
            'context_distribution': context_counts,
            'decade_distribution': decade_counts,
            'top_compounds': top_compounds,
            'foundational_papers_count': len(self.identify_foundational_papers(citations))
        }


if __name__ == "__main__":
    # Example usage
    extractor = ChemistryCitationExtractor()

    sample_text = """
    The concept of resonance was first introduced by Pauling (J. Am. Chem. Soc. 53, 1367 (1931))
    and later developed in his comprehensive work (Pauling, L. The Nature of the Chemical Bond;
    Cornell University Press: Ithaca, 1939). The Diels-Alder reaction was discovered by
    Diels and Alder (Ber. Dtsch. Chem. Ges. 62, 554 (1929)) and has since become one of the
    most important synthetic methods in organic chemistry.

    Recent studies have shown (15) that the mechanism involves a concerted process.
    This was confirmed by Brown et al. (1962) in their detailed kinetic analysis.
    """

    citations = extractor.extract_citations_from_text(sample_text)

    print(f"Extracted {len(citations)} citations:")
    for citation in citations:
        print(f"- {citation.citation_text}")
        print(f"  Type: {citation.citation_type}")
        print(f"  Priority: {citation.retrieval_priority}")
        print(f"  Score: {citation.importance_score:.2f}")
        print(f"  Context: {citation.academic_context}")
        print(f"  Entities: {citation.entities_mentioned}")
        print()

    # Generate summary
    summary = extractor.generate_citation_summary(citations)
    print("Citation Summary:")
    print(json.dumps(summary, indent=2))