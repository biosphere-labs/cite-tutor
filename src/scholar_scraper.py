"""
Google Scholar scraper for academic DOI lookup.
Specialized for finding foundational academic papers with robust rate limiting.
Supports multiple academic domains through configuration.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import random
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse
import json
from pathlib import Path
from dataclasses import dataclass
from domain_config import get_domain_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ScholarResult:
    """Data class for Scholar search results."""
    title: str
    authors: str
    publication: str
    year: Optional[int]
    url: str
    doi: Optional[str]
    cited_by: int
    validation_score: float


class ScholarScraper:
    """
    Google Scholar scraper optimized for academic paper DOI lookup.
    Handles rate limiting and provides domain-specific validation.
    Supports multiple academic domains through configuration.
    """

    # User agents for rotation
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
    ]

    # Domain configuration will be loaded on initialization

    def __init__(self, delay_range: Tuple[int, int] = (2, 5), cache_dir: str = "data/cache", domain: str = None):
        self.delay_range = delay_range
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delays = [10, 30, 60, 120, 300]  # Progressive delays
        self.current_delay_index = 0

        # Load domain configuration
        self.domain_config = get_domain_config()
        if domain:
            self.domain_config.set_domain(domain)

        logger.info(f"Scholar scraper initialized for domain: {self.domain_config.current_domain}")

    def expand_journal_abbreviations(self, citation: str) -> str:
        """Expand journal abbreviations for better search results."""
        return self.domain_config.expand_journal_abbreviations(citation)

    def clean_citation_for_search(self, citation_text: str, context: str = "") -> List[str]:
        """Generate multiple search queries for academic citations."""

        queries = []

        # Strategy 1: Expand journal abbreviations
        expanded = self.expand_journal_abbreviations(citation_text)
        queries.append(expanded)

        # Strategy 2: Extract author + year pattern
        author_year = self._extract_author_year(citation_text)
        if author_year:
            queries.append(author_year)

        # Strategy 3: Extract journal + year + volume
        journal_info = self._extract_journal_info(citation_text)
        if journal_info:
            queries.append(journal_info)

        # Strategy 4: Use context to guess title
        if context:
            title_guess = self._guess_title_from_context(context)
            if title_guess and author_year:
                queries.append(f'"{title_guess}" {author_year}')

        # Strategy 5: Clean minimal query (fallback)
        clean_query = re.sub(r'[^\w\s\-\.]', ' ', citation_text)
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        if clean_query and clean_query not in queries:
            queries.append(clean_query)

        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query and query.lower() not in seen:
                unique_queries.append(query)
                seen.add(query.lower())

        logger.info(f"Generated {len(unique_queries)} search queries for: {citation_text[:50]}...")
        return unique_queries

    def _extract_author_year(self, citation: str) -> Optional[str]:
        """Extract author and year from citation."""

        # Pattern for author-year format like "(Brown et al., 1962)"
        author_year_match = re.search(r'\(([A-Z][a-z]+(?:\s+et al\.?)?),?\s*(\d{4})\)', citation)
        if author_year_match:
            return f"{author_year_match.group(1)} {author_year_match.group(2)}"

        # Pattern for journal citation with year at end
        journal_year_match = re.search(r'([A-Z][a-z]+(?:,?\s*[A-Z]\.?)*)[^()]*\((\d{4})\)', citation)
        if journal_year_match:
            return f"{journal_year_match.group(1)} {journal_year_match.group(2)}"

        # Pattern for book citation
        book_match = re.search(r'([A-Z][a-z]+,\s*[A-Z]\.?)[^;]*;[^,]*,\s*(\d{4})', citation)
        if book_match:
            return f"{book_match.group(1)} {book_match.group(2)}"

        return None

    def _extract_journal_info(self, citation: str) -> Optional[str]:
        """Extract journal, volume, year information."""

        # Pattern: Journal Volume, Page (Year)
        journal_match = re.search(r'([A-Z][a-z]*\.?\s*(?:[A-Z][a-z]*\.?\s*)*)\s*(\d{1,4}),?\s*\d+\s*\((\d{4})\)', citation)
        if journal_match:
            journal = journal_match.group(1).strip()
            volume = journal_match.group(2)
            year = journal_match.group(3)

            # Expand abbreviations
            journal_expanded = self.expand_chemistry_journal_abbreviations(journal)
            return f"{journal_expanded} {year} volume {volume}"

        return None

    def _guess_title_from_context(self, context: str) -> Optional[str]:
        """Attempt to guess paper title from citation context."""

        # Look for phrases that might be titles
        title_patterns = [
            r'"([^"]{10,100})"',  # Quoted text
            r'titled\s+"([^"]{10,100})"',  # "titled ..."
            r'entitled\s+"([^"]{10,100})"',  # "entitled ..."
            r'study\s+of\s+([^.]{10,80})',  # "study of ..."
            r'theory\s+of\s+([^.]{10,80})',  # "theory of ..."
        ]

        for pattern in title_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Filter out overly generic titles
                if len(title.split()) >= 3 and not title.lower().startswith('the reaction'):
                    return title

        return None

    def handle_rate_limiting(self):
        """Handle rate limiting with progressive delays."""

        if self.current_delay_index < len(self.rate_limit_delays):
            delay = self.rate_limit_delays[self.current_delay_index]
            logger.warning(f"Rate limited. Waiting {delay} seconds...")
            time.sleep(delay)
            self.current_delay_index += 1
        else:
            # Maximum delay reached, wait longer
            delay = 600  # 10 minutes
            logger.warning(f"Maximum rate limit reached. Waiting {delay} seconds...")
            time.sleep(delay)

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""

        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_delay = random.uniform(*self.delay_range)

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def search_with_rotation(self, query: str) -> List[ScholarResult]:
        """Search Scholar with user agent rotation and rate limiting."""

        self._enforce_rate_limit()

        headers = {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        encoded_query = quote(query)
        url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_sdt=0%2C5"

        try:
            logger.info(f"Searching Scholar for: {query}")
            response = self.session.get(url, headers=headers, timeout=30)

            if response.status_code == 429:  # Rate limited
                logger.warning("Received 429 status code (rate limited)")
                self.handle_rate_limiting()
                return self.search_with_rotation(query)  # Retry after delay

            elif response.status_code != 200:
                logger.error(f"Scholar request failed with status {response.status_code}")
                return []

            # Reset delay index on successful request
            self.current_delay_index = 0

            return self._parse_scholar_results(response.text, query)

        except requests.exceptions.RequestException as e:
            logger.error(f"Scholar search failed: {e}")
            return []

    def _parse_scholar_results(self, html: str, original_query: str) -> List[ScholarResult]:
        """Parse Scholar HTML results."""

        soup = BeautifulSoup(html, 'html.parser')
        results = []

        # Find all result divs
        result_divs = soup.find_all('div', {'class': 'gs_r gs_or gs_scl'}) or soup.find_all('div', {'data-lid': True})

        for div in result_divs:
            try:
                result = self._parse_single_result(div)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue

        logger.info(f"Parsed {len(results)} results from Scholar")
        return results

    def _parse_single_result(self, div) -> Optional[ScholarResult]:
        """Parse a single Scholar result."""

        # Extract title
        title_elem = div.find('h3', {'class': 'gs_rt'}) or div.find('a')
        if not title_elem:
            return None

        title = title_elem.get_text().strip()
        url = title_elem.find('a')['href'] if title_elem.find('a') else ""

        # Extract authors and publication info
        author_elem = div.find('div', {'class': 'gs_a'})
        authors = ""
        publication = ""
        year = None

        if author_elem:
            author_text = author_elem.get_text()
            # Try to extract year
            year_match = re.search(r'(\d{4})', author_text)
            if year_match:
                year = int(year_match.group(1))

            # Split on year or dash to separate authors from publication
            parts = re.split(r'\d{4}|\s-\s', author_text, 1)
            if parts:
                authors = parts[0].strip()
                if len(parts) > 1:
                    publication = parts[1].strip()

        # Extract citation count
        cited_by = 0
        cited_elem = div.find('a', string=re.compile(r'Cited by \d+'))
        if cited_elem:
            cited_match = re.search(r'Cited by (\d+)', cited_elem.get_text())
            if cited_match:
                cited_by = int(cited_match.group(1))

        # Try to extract DOI
        doi = self._extract_doi_from_result(div, url)

        return ScholarResult(
            title=title,
            authors=authors,
            publication=publication,
            year=year,
            url=url,
            doi=doi,
            cited_by=cited_by,
            validation_score=0.0  # Will be calculated later
        )

    def _extract_doi_from_result(self, div, url: str) -> Optional[str]:
        """Extract DOI from Scholar result."""

        # Look for DOI in links
        all_links = div.find_all('a')
        for link in all_links:
            href = link.get('href', '')

            # Check for direct DOI links
            doi_match = re.search(r'(?:doi\.org/|doi:)(10\.\d{4,}/[^\s]+)', href)
            if doi_match:
                return doi_match.group(1)

            # Check for publisher links that might redirect to DOI
            if any(domain in href for domain in ['sciencedirect.com', 'springer.com', 'wiley.com', 'acs.org']):
                try:
                    # Follow redirect to get potential DOI
                    response = self.session.head(href, timeout=10, allow_redirects=True)
                    final_url = response.url
                    doi_match = re.search(r'(?:doi\.org/|doi:)(10\.\d{4,}/[^\s]+)', final_url)
                    if doi_match:
                        return doi_match.group(1)
                except:
                    pass

        return None

    def validate_domain_paper_match(self, result: ScholarResult, original_citation: str, context: str = "") -> float:
        """Validate that Scholar result matches the domain citation."""

        score = 0.0

        # Title domain relevance (0-0.3)
        title_relevance = self.domain_config.validate_domain_relevance(result.title)
        score += title_relevance * 0.3

        # Publication domain relevance (0-0.2)
        pub_score = self.domain_config.get_journal_impact_score(result.publication)
        score += pub_score * 0.2

        # Author matching (0-0.2)
        if result.authors:
            original_authors = self._extract_authors_from_citation(original_citation)
            if original_authors:
                author_match_score = self._calculate_author_similarity(result.authors, original_authors)
                score += author_match_score * 0.2

        # Year matching (0-0.2)
        original_year = self._extract_year_from_citation(original_citation)
        if result.year and original_year:
            year_diff = abs(result.year - original_year)
            if year_diff == 0:
                score += 0.2
            elif year_diff <= 1:
                score += 0.15
            elif year_diff <= 2:
                score += 0.1

        # Citation count bonus (0-0.1) - higher citations = more likely correct
        if result.cited_by > 0:
            citation_bonus = min(0.1, result.cited_by / 1000)
            score += citation_bonus

        return min(1.0, score)

    def _extract_authors_from_citation(self, citation: str) -> List[str]:
        """Extract author names from citation text."""

        authors = []

        # Pattern 1: "Author, A." format
        author_matches = re.findall(r'([A-Z][a-z]+),\s*[A-Z]\.?', citation)
        authors.extend(author_matches)

        # Pattern 2: "Author et al." format
        et_al_match = re.search(r'([A-Z][a-z]+)\s+et al\.?', citation)
        if et_al_match:
            authors.append(et_al_match.group(1))

        return authors

    def _extract_year_from_citation(self, citation: str) -> Optional[int]:
        """Extract year from citation."""

        year_match = re.search(r'\((\d{4})\)|,\s*(\d{4})', citation)
        if year_match:
            return int(year_match.group(1) or year_match.group(2))
        return None

    def _calculate_author_similarity(self, result_authors: str, citation_authors: List[str]) -> float:
        """Calculate similarity between result authors and citation authors."""

        result_lower = result_authors.lower()
        matches = 0

        for author in citation_authors:
            if author.lower() in result_lower:
                matches += 1

        return matches / len(citation_authors) if citation_authors else 0.0

    def search_citation(self, citation_text: str, context: str = "") -> List[ScholarResult]:
        """Main method to search for a citation and return validated results."""

        # Generate search queries
        queries = self.clean_citation_for_search(citation_text, context)

        all_results = []

        for query in queries:
            try:
                results = self.search_with_rotation(query)

                # Validate each result
                for result in results:
                    result.validation_score = self.validate_domain_paper_match(result, citation_text, context)

                all_results.extend(results)

                # If we found good results, we can stop searching
                good_results = [r for r in results if r.validation_score > 0.6]
                if good_results:
                    logger.info(f"Found {len(good_results)} good matches, stopping search")
                    break

            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                continue

        # Remove duplicates based on title similarity
        unique_results = self._deduplicate_results(all_results)

        # Sort by validation score
        unique_results.sort(key=lambda x: x.validation_score, reverse=True)

        logger.info(f"Found {len(unique_results)} unique results for citation")
        return unique_results

    def _deduplicate_results(self, results: List[ScholarResult]) -> List[ScholarResult]:
        """Remove duplicate results based on title similarity."""

        unique_results = []
        seen_titles = set()

        for result in results:
            # Normalize title for comparison
            normalized_title = re.sub(r'\W+', '', result.title.lower())

            # Check for similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_titles.add(normalized_title)

        return unique_results

    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two normalized titles are similar."""

        # Simple similarity check
        if len(title1) == 0 or len(title2) == 0:
            return False

        # Calculate overlap
        set1 = set(title1)
        set2 = set(title2)

        overlap = len(set1 & set2)
        union = len(set1 | set2)

        similarity = overlap / union if union > 0 else 0
        return similarity > 0.8

    def extract_doi_from_results(self, results: List[ScholarResult]) -> Optional[str]:
        """Extract the most reliable DOI from search results."""

        # Find the best result with a DOI
        best_result_with_doi = None
        best_score = 0.0

        for result in results:
            if result.doi and result.validation_score > best_score:
                best_result_with_doi = result
                best_score = result.validation_score

        if best_result_with_doi:
            logger.info(f"Found DOI: {best_result_with_doi.doi} (score: {best_score:.2f})")
            return best_result_with_doi.doi

        logger.warning("No DOI found in search results")
        return None

    def batch_search_citations(self, citations: List[Dict], output_path: str = None) -> Dict:
        """Search multiple citations and save results."""

        results = {}

        logger.info(f"Starting batch search for {len(citations)} citations")

        for i, citation in enumerate(citations):
            citation_text = citation.get('citation_text', '')
            context = citation.get('context', '')

            logger.info(f"Processing citation {i+1}/{len(citations)}: {citation_text[:50]}...")

            try:
                search_results = self.search_citation(citation_text, context)

                # Find best DOI
                best_doi = self.extract_doi_from_results(search_results)

                results[citation_text] = {
                    'citation_data': citation,
                    'search_results': [
                        {
                            'title': r.title,
                            'authors': r.authors,
                            'publication': r.publication,
                            'year': r.year,
                            'url': r.url,
                            'doi': r.doi,
                            'cited_by': r.cited_by,
                            'validation_score': r.validation_score
                        } for r in search_results
                    ],
                    'best_doi': best_doi,
                    'success': len(search_results) > 0
                }

            except Exception as e:
                logger.error(f"Failed to process citation: {e}")
                results[citation_text] = {
                    'citation_data': citation,
                    'search_results': [],
                    'best_doi': None,
                    'success': False,
                    'error': str(e)
                }

        # Save results if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Saved batch results to {output_path}")

        return results


if __name__ == "__main__":
    # Example usage
    scraper = ScholarScraper()

    # Get sample data from domain configuration
    domain_config = get_domain_config()
    sample_papers = domain_config.get_sample_data('paper')

    # Create test citations from sample data
    test_citations = [
        {
            'citation_text': 'Sample Citation 1 (1931)',
            'context': sample_papers[0]['question'] if sample_papers else 'Sample context'
        },
        {
            'citation_text': 'Sample Citation 2 (1939)',
            'context': sample_papers[1]['question'] if len(sample_papers) > 1 else 'Sample context'
        }
    ]

    for citation in test_citations:
        print(f"\nSearching for: {citation['citation_text']}")
        results = scraper.search_citation(citation['citation_text'], citation['context'])

        print(f"Found {len(results)} results:")
        for result in results[:3]:  # Show top 3
            print(f"  - {result.title}")
            print(f"    Authors: {result.authors}")
            print(f"    Year: {result.year}")
            print(f"    DOI: {result.doi}")
            print(f"    Score: {result.validation_score:.2f}")
            print()