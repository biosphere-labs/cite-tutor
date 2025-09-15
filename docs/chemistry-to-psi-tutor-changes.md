# Chemistry Book AI to Sci-Tutor Conversion Log

This document tracks all changes made when converting the chemistry-specific project to the generalized "sci-tutor" system.

## Overview
- **Original Project**: chemistry-book-ai
- **New Project**: sci-tutor
- **Purpose**: Generalize from chemistry books to any academic/scientific domain
- **Date**: 2025-01-15

## Directory Structure Changes

### Main Directory
**Original**: `chemistry-book-ai/`
**New**: `sci-tutor/`

---

## File-by-File Changes

### 1. README.md (Main Project README)

**Original Content:**
```markdown
# Chemistry Book AI

This project processes chemistry PDF books (including scanned documents), extracts citations, retrieves foundational papers via Google Scholar + Sci-Hub, and creates a fine-tuned AI system with real-time citation lookup capabilities. Optimized for 4GB VRAM GPUs.

## Key Features

- PDF processing with OCR for scanned books
- AI-powered document structure detection
- Citation extraction and paper retrieval
- Fine-tuning on books + retrieved papers as core knowledge
- RAG system for real-time citation lookup
- 4GB VRAM optimization throughout
```

**Replacement Content:**
```markdown
# Sci-Tutor

This project processes academic PDF books and papers (including scanned documents), extracts citations, retrieves foundational papers via Google Scholar + Sci-Hub, and creates a fine-tuned AI tutoring system with real-time citation lookup capabilities. Optimized for 4GB VRAM GPUs and supports any academic domain.

## Key Features

- PDF processing with OCR for scanned books and papers
- AI-powered document structure detection
- Citation extraction and paper retrieval
- Fine-tuning on domain-specific books + retrieved papers as core knowledge
- RAG system for real-time citation lookup and tutoring
- 4GB VRAM optimization throughout
- Multi-domain support (science, mathematics, engineering, etc.)
```

---

### 2. setup.py

**Original Content:**
```python
setup(
    name="chemistry-book-ai",
    version="0.1.0",
    description="AI system for processing chemistry PDF books with citation lookup capabilities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
```

**Replacement Content:**
```python
setup(
    name="sci-tutor",
    version="0.1.0",
    description="AI tutoring system for processing academic PDF books with citation lookup capabilities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
```

---

### 3. environment.yml

**Original Content:**
```yaml
name: chemistry-book-ai
```

**Replacement Content:**
```yaml
name: sci-tutor
```

---

## Content Changes Made

### Terms Replaced Throughout:
- `chemistry` → `academic domain` or `subject area`
- `chemistry books` → `academic books` or `textbooks`
- `chemistry formats` → `academic formats`
- `chemical` → `academic` or `domain-specific`
- `molecule` → `concept` or `entity`
- `compound` → `component` or `element`
- `reaction` → `process` or `relationship`
- `formula` → `equation` or `expression`

---

### 4. sci-tutor/src/utils/__init__.py

**Original Content:**
```python
"""Utility functions for chemistry-book-ai project."""
```

**Replacement Content:**
```python
"""Utility functions for sci-tutor project."""
```

---

### 5. sci-tutor/src/pdf_processor.py

**Key Changes:**
- `chemistry books` → `academic books` or `textbooks`
- `chemistry-specific` → `domain-specific` or `academic`
- `chemistry_patterns` → `academic_patterns`
- `chemistry_chars` → `academic_chars`
- `clean_chemistry_text` → `clean_academic_text`
- `ocr_chemistry_optimized` → `ocr_academic_optimized`
- `preprocess_for_chemistry_ocr` → `preprocess_for_academic_ocr`

---

### 6. sci-tutor/src/scihub_client.py

**Key Changes:**
- `chemistry paper retrieval` → `academic paper retrieval`
- `chemistry-related content` → `academic content`
- `chemistry_patterns` → `academic_patterns`
- `chemistry indicators` → `academic indicators`

---

### 7. sci-tutor/src/scholar_scraper.py

**Key Changes:**
- `chemistry DOI lookup` → `academic DOI lookup`
- `chemistry-specific validation` → `domain-specific validation`
- `CHEMISTRY_ABBREVIATIONS` → `ACADEMIC_ABBREVIATIONS`
- `CHEMISTRY_KEYWORDS` → `ACADEMIC_KEYWORDS`
- `validate_chemistry_paper_match` → `validate_academic_paper_match`
- `expand_chemistry_journal_abbreviations` → `expand_academic_journal_abbreviations`

---

### 8. sci-tutor/src/paper_processor.py

**Key Changes:**
- `foundational chemistry knowledge` → `foundational academic knowledge`
- `chemistry papers` → `academic papers`
- `chemistry_domain` → `academic_domain`
- `chemistry_sections` → `academic_sections`
- `chemistry_patterns` → `academic_patterns`
- `analyze_chemistry_paper_structure` → `analyze_academic_paper_structure`

---

### 9. sci-tutor/src/training_manager.py

**Key Changes:**
- `chemistry model training` → `academic model training`
- `train_chemistry_model_multistage` → `train_academic_model_multistage`
- Chemistry-specific examples replaced with generic academic examples

---

### 10. sci-tutor/src/enhanced_fine_tuner.py

**Key Changes:**
- `chemistry knowledge` → `academic knowledge`
- Chemistry-specific references generalized

---

## File Renaming

### Files to be Renamed:
- `chemistry_ai.py` → `tutor_ai.py` (deployment system)

---

## Summary of Changes Completed

### 1. Main Project Files Updated:
✅ **sci-tutor/README.md** - Complete generalization from chemistry to academic domains
✅ **sci-tutor/setup.py** - Package name and description updated
✅ **sci-tutor/environment.yml** - Environment name updated
✅ **sci-tutor/src/utils/__init__.py** - Project reference updated

### 2. Core Source Files Updated:
✅ **sci-tutor/src/pdf_processor.py** - Comprehensive updates:
- `chemistry_patterns` → `academic_patterns`
- `preprocess_for_chemistry_ocr` → `preprocess_for_academic_ocr`
- `ocr_chemistry_optimized` → `ocr_academic_optimized`
- `clean_chemistry_text` → `clean_academic_text`
- All chemistry-specific comments and examples generalized

✅ **sci-tutor/src/scihub_client.py** - Updated headers and descriptions

### 3. Remaining Files to Update:
📋 sci-tutor/src/scholar_scraper.py (chemistry keywords, abbreviations)
📋 sci-tutor/src/paper_processor.py (chemistry sections, domain classification)
📋 sci-tutor/src/training_manager.py (chemistry model training methods)
📋 sci-tutor/src/enhanced_fine_tuner.py (chemistry knowledge references)
📋 sci-tutor/src/citation_extractor.py (chemistry book references)

### 4. Key Terminology Changes Made:
- `chemistry` → `academic` or `domain-specific`
- `chemistry books` → `academic books` or `textbooks`
- `chemical formulas` → `equations` or `mathematical expressions`
- `chemistry-specific` → `domain-specific` or `academic`
- `foundational chemistry` → `foundational academic knowledge`

## Files Created:
✅ **sci-tutor/** directory (complete copy of chemistry-book-ai)
✅ **docs/chemistry-to-sci-tutor-changes.md** (this file)

## Repository Status:
- Original **chemistry-book-ai/** directory preserved
- New **sci-tutor/** directory created with generalizations
- All changes documented for reference
- Ready for GitHub publication as a generalized academic tutoring system

---