"""
Sci-Hub client for academic paper retrieval.
Handles mirror management, PDF validation, and local caching.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import hashlib
import random
from urllib.parse import urljoin, urlparse
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SciHubClient:
    """
    Sci-Hub client optimized for retrieving foundational academic papers.
    Includes mirror management, PDF validation, and intelligent caching.
    """

    # Known Sci-Hub mirrors (updated periodically as domains change)
    SCIHUB_MIRRORS = [
        "https://sci-hub.se/",
        "https://sci-hub.st/",
        "https://sci-hub.ru/",
        "https://sci-hub.tf/",
        "https://sci-hub.ren/",
        "https://sci-hub.wf/",
        "https://sci-hub.shop/",
        "https://sci-hub.ee/",
        "https://sci-hub.bz/",
        "https://sci-hub.tw/",
        "https://sci-hub.is/"
    ]

    # User agents for rotation
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (compatible; academic research bot; +http://example.com/bot)',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]

    def __init__(self, cache_dir: str = "data/papers", request_timeout: int = 60):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.working_mirror = None
        self.last_mirror_check = 0
        self.mirror_check_interval = 3600  # 1 hour

        # Chemistry content validation patterns
        self.chemistry_patterns = [
            rb'chemistry', rb'chemical', rb'molecule', rb'reaction',
            rb'synthesis', rb'organic', rb'inorganic', rb'catalyst',
            rb'spectroscopy', rb'thermodynamics', rb'kinetics'
        ]

    def find_working_mirror(self, force_refresh: bool = False) -> str:
        """Test mirrors to find currently working one with caching."""

        current_time = time.time()

        # Use cached working mirror if recent and not forcing refresh
        if (not force_refresh and
            self.working_mirror and
            (current_time - self.last_mirror_check) < self.mirror_check_interval):
            return self.working_mirror

        logger.info("Testing Sci-Hub mirrors for availability...")

        # Randomize mirror order to distribute load
        mirrors_to_test = self.SCIHUB_MIRRORS.copy()
        random.shuffle(mirrors_to_test)

        for mirror in mirrors_to_test:
            try:
                headers = {
                    'User-Agent': random.choice(self.USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }

                response = self.session.get(mirror, headers=headers, timeout=10)

                # Check if response indicates working Sci-Hub
                if (response.status_code == 200 and
                    any(keyword in response.text.lower() for keyword in ['sci-hub', 'scientific', 'research'])):

                    logger.info(f"Working mirror found: {mirror}")
                    self.working_mirror = mirror
                    self.last_mirror_check = current_time
                    return mirror

            except Exception as e:
                logger.debug(f"Mirror {mirror} failed: {e}")
                continue

        raise Exception("No working Sci-Hub mirrors found. Please check internet connection or try again later.")

    def check_local_cache(self, doi: str) -> Optional[bytes]:
        """Check if paper is already cached locally."""

        safe_filename = self._create_safe_filename(doi)
        cache_path = self.cache_dir / f"{safe_filename}.pdf"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    pdf_bytes = f.read()

                # Validate cached PDF
                if self.validate_pdf_content(pdf_bytes):
                    logger.info(f"Found cached paper for DOI: {doi}")
                    return pdf_bytes
                else:
                    logger.warning(f"Cached PDF for {doi} is corrupted, removing...")
                    cache_path.unlink()

            except Exception as e:
                logger.error(f"Error reading cached paper {doi}: {e}")

        return None

    def _create_safe_filename(self, doi: str) -> str:
        """Create safe filename from DOI."""

        # Replace problematic characters
        safe_name = re.sub(r'[^\w\-_\.]', '_', doi)

        # Limit length and add hash if too long
        if len(safe_name) > 100:
            hash_suffix = hashlib.md5(doi.encode()).hexdigest()[:8]
            safe_name = safe_name[:90] + "_" + hash_suffix

        return safe_name

    def extract_pdf_url_from_html(self, html: str, base_url: str) -> Optional[str]:
        """Extract PDF download URL from Sci-Hub HTML response."""

        soup = BeautifulSoup(html, 'html.parser')

        # Strategy 1: Look for iframe containing PDF
        iframe = soup.find('iframe', {'id': 'pdf'}) or soup.find('iframe', src=True)
        if iframe and iframe.get('src'):
            pdf_url = urljoin(base_url, iframe['src'])
            return pdf_url

        # Strategy 2: Look for direct PDF links
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
        if pdf_links:
            pdf_url = urljoin(base_url, pdf_links[0]['href'])
            return pdf_url

        # Strategy 3: Look for embed tags
        embed = soup.find('embed', type='application/pdf')
        if embed and embed.get('src'):
            pdf_url = urljoin(base_url, embed['src'])
            return pdf_url

        # Strategy 4: Look for object tags
        obj = soup.find('object', {'data': True, 'type': 'application/pdf'})
        if obj:
            pdf_url = urljoin(base_url, obj['data'])
            return pdf_url

        # Strategy 5: Search for PDF URLs in JavaScript
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                pdf_match = re.search(r'(?:https?://[^\s"\']+\.pdf)', script.string, re.IGNORECASE)
                if pdf_match:
                    return pdf_match.group(0)

        return None

    def validate_pdf_content(self, pdf_bytes: bytes) -> bool:
        """Validate that bytes represent a valid PDF with reasonable content."""

        if not pdf_bytes:
            return False

        # Check PDF header
        if not pdf_bytes.startswith(b'%PDF-'):
            logger.warning("Content does not have valid PDF header")
            return False

        # Check reasonable file size (100KB - 50MB)
        size_mb = len(pdf_bytes) / (1024 * 1024)
        if size_mb < 0.1 or size_mb > 50:
            logger.warning(f"PDF size ({size_mb:.2f}MB) is outside reasonable range")
            return False

        # Check PDF structure
        if b'%%EOF' not in pdf_bytes:
            logger.warning("PDF does not contain proper ending marker")
            return False

        # Check for some text content (not just images)
        if b'/Type/Page' not in pdf_bytes:
            logger.warning("PDF does not appear to contain pages")
            return False

        # Optional: Check for chemistry-related content in first 10KB
        # This helps filter out non-academic papers
        sample = pdf_bytes[:10240].lower()
        chemistry_indicators = sum(1 for pattern in self.chemistry_patterns if pattern in sample)

        if chemistry_indicators == 0:
            logger.info("PDF does not contain obvious chemistry indicators (may still be valid)")

        logger.info(f"PDF validation passed: {size_mb:.2f}MB, {chemistry_indicators} chemistry indicators")
        return True

    def get_paper_by_doi(self, doi: str, retry_attempts: int = 3) -> bytes:
        """
        Main method to retrieve paper PDF from Sci-Hub.
        Returns PDF bytes or raises exception.
        """

        logger.info(f"Retrieving paper with DOI: {doi}")

        # Check local cache first
        cached_pdf = self.check_local_cache(doi)
        if cached_pdf:
            return cached_pdf

        # Clean DOI format
        clean_doi = doi.strip().lstrip('doi:').lstrip('/')

        for attempt in range(retry_attempts):
            try:
                # Find working mirror
                working_mirror = self.find_working_mirror()

                # Construct URL
                paper_url = f"{working_mirror}{clean_doi}"

                headers = {
                    'User-Agent': random.choice(self.USER_AGENTS),
                    'Accept': 'application/pdf,text/html,application/xhtml+xml,*/*;q=0.9',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': working_mirror,
                    'Connection': 'keep-alive',
                }

                logger.info(f"Attempting to retrieve from: {paper_url}")

                # Initial request
                response = self.session.get(
                    paper_url,
                    headers=headers,
                    timeout=self.request_timeout,
                    stream=True
                )

                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.reason}")

                content_type = response.headers.get('content-type', '').lower()

                if 'application/pdf' in content_type:
                    # Direct PDF response
                    logger.info("Received direct PDF response")
                    pdf_bytes = response.content

                elif 'text/html' in content_type:
                    # HTML page with PDF link
                    logger.info("Received HTML response, extracting PDF URL...")
                    pdf_url = self.extract_pdf_url_from_html(response.text, working_mirror)

                    if not pdf_url:
                        raise Exception("Could not find PDF URL in HTML response")

                    logger.info(f"Found PDF URL: {pdf_url}")

                    # Request the actual PDF
                    pdf_response = self.session.get(
                        pdf_url,
                        headers=headers,
                        timeout=self.request_timeout
                    )

                    if pdf_response.status_code != 200:
                        raise Exception(f"PDF download failed: HTTP {pdf_response.status_code}")

                    pdf_bytes = pdf_response.content

                else:
                    raise Exception(f"Unexpected content type: {content_type}")

                # Validate PDF content
                if not self.validate_pdf_content(pdf_bytes):
                    raise Exception("Retrieved content is not a valid PDF")

                # Cache the paper
                metadata = {
                    'url': paper_url,
                    'content_type': content_type,
                    'retrieval_attempt': attempt + 1
                }
                self.cache_paper_locally(clean_doi, pdf_bytes, metadata)

                logger.info(f"Successfully retrieved paper: {len(pdf_bytes)} bytes")
                return pdf_bytes

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt < retry_attempts - 1:
                    # Wait before retry, force mirror refresh on last attempt
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

                    # Force mirror refresh if this was a connection issue
                    if 'timeout' in str(e).lower() or 'connection' in str(e).lower():
                        self.working_mirror = None
                else:
                    raise Exception(f"Failed to retrieve paper after {retry_attempts} attempts: {e}")

    def cache_paper_locally(self, doi: str, pdf_bytes: bytes, metadata: Dict = None) -> None:
        """Cache retrieved papers locally with metadata."""

        safe_filename = self._create_safe_filename(doi)
        cache_path = self.cache_dir / f"{safe_filename}.pdf"

        try:
            # Save PDF
            with open(cache_path, 'wb') as f:
                f.write(pdf_bytes)

            # Save metadata
            metadata_path = cache_path.with_suffix('.json')
            cache_metadata = {
                'doi': doi,
                'retrieved_date': datetime.now().isoformat(),
                'file_size': len(pdf_bytes),
                'cache_path': str(cache_path),
                'validation_passed': True,
                **(metadata or {})
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Cached paper: {cache_path}")

        except Exception as e:
            logger.error(f"Failed to cache paper {doi}: {e}")
            # Clean up partial files
            if cache_path.exists():
                cache_path.unlink()

    def get_cache_info(self) -> Dict:
        """Get information about cached papers."""

        cache_info = {
            'cache_directory': str(self.cache_dir),
            'total_papers': 0,
            'total_size_mb': 0,
            'papers': []
        }

        pdf_files = list(self.cache_dir.glob('*.pdf'))
        cache_info['total_papers'] = len(pdf_files)

        for pdf_file in pdf_files:
            try:
                # Get file info
                stat = pdf_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                cache_info['total_size_mb'] += size_mb

                # Load metadata if available
                metadata_file = pdf_file.with_suffix('.json')
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                paper_info = {
                    'filename': pdf_file.name,
                    'size_mb': round(size_mb, 2),
                    'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    **metadata
                }

                cache_info['papers'].append(paper_info)

            except Exception as e:
                logger.warning(f"Error reading cache info for {pdf_file}: {e}")

        cache_info['total_size_mb'] = round(cache_info['total_size_mb'], 2)
        return cache_info

    def cleanup_cache(self, max_age_days: int = 30, max_size_mb: int = 1000) -> Dict:
        """Clean up old or large cache files."""

        cleanup_stats = {
            'files_removed': 0,
            'space_freed_mb': 0,
            'errors': []
        }

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        # Get all PDF files with their ages and sizes
        pdf_files = []
        total_size = 0

        for pdf_file in self.cache_dir.glob('*.pdf'):
            try:
                stat = pdf_file.stat()
                age_seconds = current_time - stat.st_mtime
                size_mb = stat.st_size / (1024 * 1024)

                pdf_files.append({
                    'path': pdf_file,
                    'age_seconds': age_seconds,
                    'size_mb': size_mb
                })
                total_size += size_mb

            except Exception as e:
                cleanup_stats['errors'].append(f"Error accessing {pdf_file}: {e}")

        # Remove old files
        for file_info in pdf_files:
            if file_info['age_seconds'] > max_age_seconds:
                try:
                    file_info['path'].unlink()
                    # Also remove metadata file
                    metadata_file = file_info['path'].with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()

                    cleanup_stats['files_removed'] += 1
                    cleanup_stats['space_freed_mb'] += file_info['size_mb']
                    total_size -= file_info['size_mb']

                except Exception as e:
                    cleanup_stats['errors'].append(f"Error removing {file_info['path']}: {e}")

        # Remove largest files if still over size limit
        if total_size > max_size_mb:
            # Sort by size descending
            remaining_files = [f for f in pdf_files if f['path'].exists()]
            remaining_files.sort(key=lambda x: x['size_mb'], reverse=True)

            for file_info in remaining_files:
                if total_size <= max_size_mb:
                    break

                try:
                    file_info['path'].unlink()
                    metadata_file = file_info['path'].with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()

                    cleanup_stats['files_removed'] += 1
                    cleanup_stats['space_freed_mb'] += file_info['size_mb']
                    total_size -= file_info['size_mb']

                except Exception as e:
                    cleanup_stats['errors'].append(f"Error removing {file_info['path']}: {e}")

        cleanup_stats['space_freed_mb'] = round(cleanup_stats['space_freed_mb'], 2)
        logger.info(f"Cache cleanup completed: {cleanup_stats['files_removed']} files removed, "
                   f"{cleanup_stats['space_freed_mb']}MB freed")

        return cleanup_stats

    def batch_retrieve_papers(self, dois: List[str], max_concurrent: int = 3) -> Dict:
        """Retrieve multiple papers with rate limiting."""

        results = {}

        logger.info(f"Starting batch retrieval of {len(dois)} papers")

        for i, doi in enumerate(dois):
            logger.info(f"Processing paper {i+1}/{len(dois)}: {doi}")

            try:
                pdf_bytes = self.get_paper_by_doi(doi)
                results[doi] = {
                    'success': True,
                    'size_bytes': len(pdf_bytes),
                    'error': None
                }

                # Rate limiting between requests
                if i < len(dois) - 1:  # Don't wait after last request
                    wait_time = random.uniform(3, 7)  # 3-7 seconds
                    logger.info(f"Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Failed to retrieve {doi}: {e}")
                results[doi] = {
                    'success': False,
                    'size_bytes': 0,
                    'error': str(e)
                }

        successful = sum(1 for r in results.values() if r['success'])
        logger.info(f"Batch retrieval completed: {successful}/{len(dois)} papers retrieved successfully")

        return results


if __name__ == "__main__":
    # Example usage
    client = SciHubClient()

    # Test with a chemistry paper DOI
    test_doi = "10.1021/ja01367a002"  # Example chemistry DOI

    try:
        print(f"Testing Sci-Hub client with DOI: {test_doi}")

        # Test mirror finding
        working_mirror = client.find_working_mirror()
        print(f"Working mirror: {working_mirror}")

        # Test paper retrieval
        pdf_bytes = client.get_paper_by_doi(test_doi)
        print(f"Retrieved paper: {len(pdf_bytes)} bytes")

        # Check cache info
        cache_info = client.get_cache_info()
        print(f"Cache info: {cache_info['total_papers']} papers, {cache_info['total_size_mb']}MB")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This is expected if no working Sci-Hub mirrors are available")