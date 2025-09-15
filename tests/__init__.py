"""
Test package for sci-tutor application.

This package contains comprehensive unit and integration tests for all components
of the sci-tutor system. Tests are designed with mock abstraction to allow
switching between unit tests (mocked) and integration tests (real calls).

Test Structure:
- conftest.py: Shared fixtures and configuration
- test_*.py: Individual test modules for each component
- mocks/: Mock implementations for external dependencies
- fixtures/: Test data and fixtures
- utils/: Test utilities and helpers

Environment Variables:
- TEST_MODE: 'unit' or 'integration' (default: 'unit')
- MOCK_EXTERNAL_APIS: 'true' or 'false' (default: 'true')
- TESTING: Always 'true' during test runs

Usage:
    # Run all unit tests (default)
    pytest

    # Run integration tests (requires network access)
    TEST_MODE=integration pytest -m integration

    # Run specific test file
    pytest tests/test_pdf_processor.py

    # Run with coverage
    pytest --cov=src --cov-report=html
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Test configuration
TEST_MODE = os.getenv("TEST_MODE", "unit")
MOCK_EXTERNAL_APIS = os.getenv("MOCK_EXTERNAL_APIS", "true").lower() == "true"
TESTING = True

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_DATA_DIR.mkdir(exist_ok=True)