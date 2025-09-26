"""
Cite-Tutor: AI-powered academic research assistant.

This package provides tools for processing PDF books and papers, extracting citations,
retrieving foundational papers, and creating fine-tuned AI tutoring systems with
real-time citation lookup capabilities.

Optimized for 4GB VRAM GPUs and supports multiple academic domains.
"""

__version__ = "0.1.0"
__author__ = "Justin Robinson"
__email__ = "justin.g.robinson@gmail.com"

from cite_tutor.domain_config import DomainConfiguration
from cite_tutor.pdf_processor import PDFProcessor
from cite_tutor.citation_extractor import CitationExtractor
from cite_tutor.paper_processor import PaperProcessor
from cite_tutor.enhanced_fine_tuner import EnhancedFineTuner

__all__ = [
    "DomainConfiguration",
    "PDFProcessor",
    "CitationExtractor",
    "PaperProcessor",
    "EnhancedFineTuner",
]