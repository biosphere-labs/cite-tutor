"""
Robust PDF text extraction module optimized for academic books.
Handles both modern PDFs and scanned documents with domain-specific processing.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Robust PDF text extraction for academic books with OCR support.
    Optimized for both modern PDFs and scanned historical documents.
    """

    def __init__(self, cache_dir: str = "data/cache", enable_caching: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_caching = enable_caching
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Academic patterns for validation (science, math, engineering)
        self.academic_patterns = {
            'equations': re.compile(r'\b([A-Z][a-z]?\d*)+\b|[xy]\s*=|f\([xyz]\)|∂|∇|∫|∑'),
            'relationships': re.compile(r'therefore|thus|implies|follows|given|where|such that'),
            'processes': re.compile(r'→|←|↔|⇌|yields?|results?|leads to|causes|transforms'),
            'terms': re.compile(r'\b(?:theorem|proof|hypothesis|analysis|method|principle|'
                               r'theory|law|equation|function|variable|constant|parameter|'
                               r'model|system|process|structure)\b', re.IGNORECASE),
            'units': re.compile(r'\b(?:kg|m|s|A|K|mol|cd|Hz|N|Pa|J|W|C|V|F|Ω|S|Wb|T|H|°C|°F)\b'),
        }

        # Common OCR artifacts to clean
        self.ocr_artifacts = [
            (r'\s+', ' '),  # Multiple spaces
            (r'([a-z])([A-Z])', r'\1 \2'),  # Missing spaces between words
            (r'(\d)([A-Z])', r'\1 \2'),  # Numbers attached to letters
            (r'([a-z])(\d)', r'\1\2'),  # Keep equations and formulas intact
            (r'\s*-\s*\n\s*', ''),  # Hyphenated line breaks
            (r'\n{3,}', '\n\n'),  # Multiple line breaks
        ]

    def _get_cache_key(self, pdf_path: str) -> str:
        """Generate cache key based on file path and modification time."""
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            return None

        # Include file size and modification time in hash
        stat = path_obj.stat()
        cache_data = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load extraction results from cache."""
        if not self.enable_caching or not cache_key:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save extraction results to cache."""
        if not self.enable_caching or not cache_key:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def detect_if_scanned(self, pdf_path: str) -> bool:
        """
        Detect if PDF is scanned by analyzing text extraction success.
        Returns True if document appears to be scanned (needs OCR).
        """
        try:
            doc = fitz.open(pdf_path)
            total_chars = 0
            pages_checked = min(3, len(doc))  # Check first 3 pages

            for page_num in range(pages_checked):
                page = doc[page_num]
                text = page.get_text()
                total_chars += len(text.strip())

            doc.close()

            # If very little text extracted, likely scanned
            avg_chars_per_page = total_chars / pages_checked if pages_checked > 0 else 0
            is_scanned = avg_chars_per_page < 100

            logger.info(f"PDF scan detection: {avg_chars_per_page:.0f} chars/page, scanned: {is_scanned}")
            return is_scanned

        except Exception as e:
            logger.error(f"Error detecting scan status: {e}")
            return True  # Assume scanned if detection fails

    def preprocess_for_academic_ocr(self, img: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR of academic text and equations.
        """
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        # Enhance contrast for better text recognition
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Sharpen image
        img = img.filter(ImageFilter.SHARPEN)

        # Resize for better OCR (optimal DPI around 300)
        width, height = img.size
        if width < 1000:  # Upscale small images
            scale_factor = 1000 / width
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)

        return img

    def ocr_academic_optimized(self, pdf_path: str) -> str:
        """OCR optimized for academic text with equations and symbols."""
        # Extended character whitelist for academic content
        academic_chars = (
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            '()[]{}+-=→←↔.,;:\'\" \n\t/\\%°ασβπδγλμνρστφχψω'
        )
        ocr_config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={academic_chars}'

        try:
            doc = fitz.open(pdf_path)
            ocr_text = ""

            logger.info(f"Starting OCR for {len(doc)} pages")

            with tqdm(total=len(doc), desc="OCR Processing") as pbar:
                for page_num, page in enumerate(doc):
                    try:
                        # Higher resolution for better formula recognition
                        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))

                        # Preprocess image for better OCR
                        img = self.preprocess_for_academic_ocr(img)
                        page_text = pytesseract.image_to_string(img, config=ocr_config)

                        if len(page_text.strip()) > 20:
                            ocr_text += f"\n--- Page {page_num + 1} ---\n" + page_text

                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                        continue

                    pbar.update(1)

            doc.close()
            return ocr_text

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ""

    def extract_text_direct(self, pdf_path: str) -> str:
        """Extract text directly from PDF without OCR."""
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n" + page_text

            doc.close()
            return text

        except Exception as e:
            logger.error(f"Direct text extraction failed: {e}")
            return ""

    def clean_academic_text(self, raw_text: str) -> str:
        """
        Clean text while preserving equations, formulas and academic notation.
        """
        text = raw_text

        # Apply OCR artifact corrections
        for pattern, replacement in self.ocr_artifacts:
            text = re.sub(pattern, replacement, text)

        # Preserve common academic symbols that might be corrupted
        academic_fixes = [
            # Common OCR errors in academic texts
            (r'\bH20\b', 'H2O'),  # Zero instead of O
            (r'\bC02\b', 'CO2'),
            (r'\b([A-Z][a-z]?)(\s+)(\d+)', r'\1\3'),  # Fix separated subscripts
            (r'(\d+)\s+([A-Z][a-z]?)\b', r'\1\2'),  # Fix separated coefficients
            (r'([a-z])\s*-\s*([a-z])', r'\1-\2'),  # Fix separated bonds
            # Fix reaction arrows
            (r'->', '→'),
            (r'<-', '←'),
            (r'<->', '↔'),
            # Clean up whitespace around formulas
            (r'([A-Z][a-z]?\d*)\s+([A-Z][a-z]?\d*)', r'\1\2'),  # H 2 O -> H2O
        ]

        for pattern, replacement in academic_fixes:
            text = re.sub(pattern, replacement, text)

        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = text.strip()

        return text

    def estimate_extraction_quality(self, text: str) -> float:
        """
        Estimate text extraction quality for academic content.
        Returns score from 0.0 to 1.0.
        """
        if not text or len(text.strip()) < 50:
            return 0.0

        score = 0.0
        total_checks = 0

        # Check for academic content
        for pattern_name, pattern in self.academic_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                score += min(matches / 10, 1.0)  # Cap at 1.0 per pattern
            total_checks += 1

        # Normalize by number of pattern types
        if total_checks > 0:
            score = score / total_checks

        # Penalty for obvious OCR errors
        error_patterns = [
            r'[^\w\s\-\(\)\[\]\{\}→←↔.,;:\'\"\/\\%°ασβπδγλμνρστφχψω]',  # Strange characters
            r'\b[a-z]{1}[A-Z]{1}[a-z]{1}\b',  # Mixed case words
            r'\d{5,}',  # Very long numbers (likely OCR errors)
        ]

        for pattern in error_patterns:
            errors = len(re.findall(pattern, text))
            if errors > 0:
                penalty = min(errors / len(text.split()) * 10, 0.3)
                score -= penalty

        # Text length bonus (longer text usually means better extraction)
        length_bonus = min(len(text) / 10000, 0.2)
        score += length_bonus

        return max(0.0, min(1.0, score))

    def extract_text_robust(self, pdf_path: str) -> str:
        """
        Main extraction method using best approach for the document.
        Returns the highest quality extraction available.
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check cache first
        cache_key = self._get_cache_key(pdf_path)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Loaded from cache: {pdf_path}")
            return cached_result['text']

        logger.info(f"Processing PDF: {pdf_path}")

        # Try different extraction methods
        methods = {}

        # Method 1: Direct text extraction
        logger.info("Trying direct text extraction...")
        direct_text = self.extract_text_direct(pdf_path)
        if direct_text:
            direct_quality = self.estimate_extraction_quality(direct_text)
            methods['direct'] = {
                'text': self.clean_academic_text(direct_text),
                'quality': direct_quality,
                'method': 'direct'
            }
            logger.info(f"Direct extraction quality: {direct_quality:.2f}")

        # Method 2: OCR if document appears scanned or direct extraction failed
        is_scanned = self.detect_if_scanned(pdf_path)
        if is_scanned or not direct_text or methods.get('direct', {}).get('quality', 0) < 0.3:
            logger.info("Trying OCR extraction...")
            ocr_text = self.ocr_academic_optimized(pdf_path)
            if ocr_text:
                ocr_quality = self.estimate_extraction_quality(ocr_text)
                methods['ocr'] = {
                    'text': self.clean_academic_text(ocr_text),
                    'quality': ocr_quality,
                    'method': 'ocr'
                }
                logger.info(f"OCR extraction quality: {ocr_quality:.2f}")

        # Method 3: Hybrid approach (combine both if available)
        if len(methods) == 2:
            hybrid_text = methods['direct']['text'] + "\n\n--- OCR SUPPLEMENT ---\n\n" + methods['ocr']['text']
            hybrid_quality = (methods['direct']['quality'] + methods['ocr']['quality']) / 2
            methods['hybrid'] = {
                'text': self.clean_academic_text(hybrid_text),
                'quality': hybrid_quality,
                'method': 'hybrid'
            }
            logger.info(f"Hybrid extraction quality: {hybrid_quality:.2f}")

        # Choose best method
        if not methods:
            raise Exception(f"All extraction methods failed for {pdf_path}")

        best_method = max(methods.keys(), key=lambda k: methods[k]['quality'])
        result = methods[best_method]

        logger.info(f"Best extraction method: {best_method} (quality: {result['quality']:.2f})")

        # Cache result
        cache_data = {
            'text': result['text'],
            'quality': result['quality'],
            'method': result['method'],
            'all_methods': {k: {'quality': v['quality'], 'method': v['method']} for k, v in methods.items()}
        }
        self._save_to_cache(cache_key, cache_data)

        return result['text']

    def batch_extract(self, pdf_paths: List[str], output_dir: str = "data/extracted") -> Dict[str, Dict]:
        """
        Extract text from multiple PDFs with progress tracking.
        Returns dict with extraction results and metadata.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        with tqdm(total=len(pdf_paths), desc="Batch PDF Processing") as pbar:
            for pdf_path in pdf_paths:
                try:
                    pdf_name = Path(pdf_path).stem

                    # Extract text
                    extracted_text = self.extract_text_robust(pdf_path)
                    quality = self.estimate_extraction_quality(extracted_text)

                    # Save extracted text
                    text_file = output_path / f"{pdf_name}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)

                    results[pdf_path] = {
                        'success': True,
                        'quality': quality,
                        'text_length': len(extracted_text),
                        'output_file': str(text_file),
                        'error': None
                    }

                    logger.info(f"Completed: {pdf_name} (quality: {quality:.2f})")

                except Exception as e:
                    results[pdf_path] = {
                        'success': False,
                        'quality': 0.0,
                        'text_length': 0,
                        'output_file': None,
                        'error': str(e)
                    }
                    logger.error(f"Failed to process {pdf_path}: {e}")

                pbar.update(1)

        # Save batch results summary
        summary_file = output_path / "batch_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PDF Extraction Batch Summary\n")
            f.write("=" * 50 + "\n\n")

            successful = sum(1 for r in results.values() if r['success'])
            f.write(f"Total PDFs processed: {len(pdf_paths)}\n")
            f.write(f"Successful extractions: {successful}\n")
            f.write(f"Failed extractions: {len(pdf_paths) - successful}\n\n")

            for pdf_path, result in results.items():
                f.write(f"\nFile: {Path(pdf_path).name}\n")
                f.write(f"  Success: {result['success']}\n")
                f.write(f"  Quality: {result['quality']:.2f}\n")
                f.write(f"  Text Length: {result['text_length']}\n")
                if result['error']:
                    f.write(f"  Error: {result['error']}\n")

        return results


if __name__ == "__main__":
    # Example usage
    extractor = PDFExtractor()

    # Test with a single PDF
    try:
        pdf_file = "data/books/sample_academic_book.pdf"
        if Path(pdf_file).exists():
            text = extractor.extract_text_robust(pdf_file)
            quality = extractor.estimate_extraction_quality(text)
            print(f"Extracted {len(text)} characters with quality {quality:.2f}")
            print(f"First 500 characters:\n{text[:500]}")
        else:
            print(f"Sample PDF not found: {pdf_file}")
            print("Place academic PDF books in data/books/ directory to test")

    except Exception as e:
        print(f"Error: {e}")