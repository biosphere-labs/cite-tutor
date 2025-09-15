"""
Domain Configuration Manager for Cite-Tutor

This module provides a centralized way to access domain-specific configurations,
replacing hardcoded domain-specific patterns and data throughout the application.
"""

import yaml
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DomainConfigurationError(Exception):
    """Exception raised for domain configuration errors."""
    pass


class DomainConfiguration:
    """
    Centralized domain configuration management.

    This class loads and provides access to domain-specific configurations
    including journals, keywords, authors, patterns, and sample data.
    """

    def __init__(self, config_path: str = None, default_domain: str = None):
        """
        Initialize domain configuration.

        Args:
            config_path: Path to domains.yaml configuration file
            default_domain: Default domain to use if not specified
        """
        if config_path is None:
            # Default to config/domains.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "domains.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.current_domain = default_domain or self.config.get('default_domain', 'chemistry')

        # Validate domain
        if self.current_domain not in self.get_available_domains():
            raise DomainConfigurationError(f"Domain '{self.current_domain}' not found in configuration")

        # Cache compiled regex patterns
        self._compiled_patterns = {}

        logger.info(f"Domain configuration loaded. Current domain: {self.current_domain}")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise DomainConfigurationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise DomainConfigurationError(f"Error parsing configuration file: {e}")

    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return self.config.get('available_domains', [])

    def set_domain(self, domain: str):
        """
        Set the current active domain.

        Args:
            domain: Domain name to set as active

        Raises:
            DomainConfigurationError: If domain is not available
        """
        if domain not in self.get_available_domains():
            raise DomainConfigurationError(f"Domain '{domain}' not found in configuration")

        self.current_domain = domain
        # Clear compiled patterns cache when domain changes
        self._compiled_patterns = {}
        logger.info(f"Active domain changed to: {domain}")

    def get_domain_info(self, domain: str = None) -> Dict:
        """
        Get basic information about a domain.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary with domain name and description
        """
        domain = domain or self.current_domain
        domain_config = self.config['domains'].get(domain, {})

        return {
            'name': domain_config.get('name', domain.title()),
            'description': domain_config.get('description', f'{domain.title()} domain')
        }

    def get_journals(self, domain: str = None) -> Dict[str, str]:
        """
        Get journal abbreviations and full names for domain.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary mapping journal abbreviations to full names
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('journals', {})

    def get_keywords(self, domain: str = None) -> List[str]:
        """
        Get domain-specific keywords for validation.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            List of domain keywords
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('keywords', [])

    def get_foundational_authors(self, domain: str = None) -> List[str]:
        """
        Get list of foundational authors for domain.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            List of foundational author names
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('foundational_authors', [])

    def get_contexts(self, domain: str = None) -> Dict[str, List[str]]:
        """
        Get context classifications and their keywords.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary mapping context types to keyword lists
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('contexts', {})

    def get_sections(self, domain: str = None) -> Dict[str, str]:
        """
        Get document section patterns.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary mapping section names to regex patterns
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('sections', {})

    def get_patterns(self, domain: str = None) -> Dict[str, str]:
        """
        Get content patterns.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary mapping pattern names to regex patterns
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('patterns', {})

    def get_compiled_pattern(self, pattern_name: str, domain: str = None) -> re.Pattern:
        """
        Get compiled regex pattern.

        Args:
            pattern_name: Name of pattern from sections or patterns
            domain: Domain name (uses current domain if None)

        Returns:
            Compiled regex pattern

        Raises:
            DomainConfigurationError: If pattern not found
        """
        domain = domain or self.current_domain
        cache_key = f"{domain}:{pattern_name}"

        if cache_key in self._compiled_patterns:
            return self._compiled_patterns[cache_key]

        # Look in sections first, then patterns
        sections = self.get_sections(domain)
        patterns = self.get_patterns(domain)

        pattern_str = sections.get(pattern_name) or patterns.get(pattern_name)

        if not pattern_str:
            raise DomainConfigurationError(f"Pattern '{pattern_name}' not found for domain '{domain}'")

        try:
            compiled_pattern = re.compile(pattern_str, re.IGNORECASE)
            self._compiled_patterns[cache_key] = compiled_pattern
            return compiled_pattern
        except re.error as e:
            raise DomainConfigurationError(f"Invalid regex pattern '{pattern_name}': {e}")

    def get_subdomain_keywords(self, domain: str = None) -> Dict[str, List[str]]:
        """
        Get subdomain classification keywords.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary mapping subdomain names to keyword lists
        """
        domain = domain or self.current_domain
        return self.config['domains'].get(domain, {}).get('subdomain_keywords', {})

    def get_sample_data(self, data_type: str, domain: str = None) -> List[Dict]:
        """
        Get sample training data.

        Args:
            data_type: Type of sample data ('book', 'paper', 'integrated')
            domain: Domain name (uses current domain if None)

        Returns:
            List of sample Q&A pairs
        """
        domain = domain or self.current_domain
        sample_data = self.config['domains'].get(domain, {}).get('sample_data', {})
        return sample_data.get(data_type, [])

    def classify_context(self, text: str, domain: str = None) -> str:
        """
        Classify text context based on domain keywords.

        Args:
            text: Text to classify
            domain: Domain name (uses current domain if None)

        Returns:
            Most likely context type
        """
        domain = domain or self.current_domain
        contexts = self.get_contexts(domain)

        if not contexts:
            return 'general'

        text_lower = text.lower()
        scores = {}

        for context_type, keywords in contexts.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[context_type] = score

        if scores:
            return max(scores, key=scores.get)
        else:
            return 'general'

    def classify_subdomain(self, text: str, domain: str = None) -> str:
        """
        Classify text subdomain based on keywords.

        Args:
            text: Text to classify
            domain: Domain name (uses current domain if None)

        Returns:
            Most likely subdomain
        """
        domain = domain or self.current_domain
        subdomain_keywords = self.get_subdomain_keywords(domain)

        if not subdomain_keywords:
            return 'general'

        text_lower = text.lower()

        for subdomain, keywords in subdomain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return subdomain

        return 'general'

    def expand_journal_abbreviations(self, text: str, domain: str = None) -> str:
        """
        Expand journal abbreviations in text.

        Args:
            text: Text containing journal abbreviations
            domain: Domain name (uses current domain if None)

        Returns:
            Text with expanded journal names
        """
        domain = domain or self.current_domain
        journals = self.get_journals(domain)

        text_lower = text.lower()
        expanded = text

        for abbrev, full_name in journals.items():
            if abbrev in text_lower:
                # Replace abbreviation with full name (case-insensitive)
                expanded = re.sub(re.escape(abbrev), full_name, expanded, flags=re.IGNORECASE)
                break

        return expanded

    def is_foundational_author(self, author_text: str, domain: str = None) -> bool:
        """
        Check if text contains foundational authors.

        Args:
            author_text: Text to check for author names
            domain: Domain name (uses current domain if None)

        Returns:
            True if foundational author found
        """
        domain = domain or self.current_domain
        foundational_authors = self.get_foundational_authors(domain)

        author_text_lower = author_text.lower()
        return any(author in author_text_lower for author in foundational_authors)

    def validate_domain_relevance(self, text: str, domain: str = None) -> float:
        """
        Validate domain relevance of text based on keywords.

        Args:
            text: Text to validate
            domain: Domain name (uses current domain if None)

        Returns:
            Relevance score between 0 and 1
        """
        domain = domain or self.current_domain
        keywords = self.get_keywords(domain)

        if not keywords:
            return 0.5  # Neutral if no keywords defined

        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)

        # Calculate relevance score
        relevance = min(1.0, matches / max(10, len(keywords) * 0.1))
        return relevance

    def get_journal_impact_score(self, journal: str, domain: str = None) -> float:
        """
        Get impact score for journal based on domain configuration.

        Args:
            journal: Journal name or abbreviation
            domain: Domain name (uses current domain if None)

        Returns:
            Impact score between 0 and 1
        """
        domain = domain or self.current_domain
        journals = self.get_journals(domain)

        journal_lower = journal.lower().strip()

        # High impact journals
        high_impact = ['nature', 'science']
        if any(hi in journal_lower for hi in high_impact):
            return 1.0

        # Check if journal is in domain configuration
        for abbrev, full_name in journals.items():
            if abbrev in journal_lower or full_name.lower() in journal_lower:
                # Major domain journals get high score
                if any(word in abbrev for word in ['j. am.', 'proc. natl.', 'rev.']):
                    return 0.9
                else:
                    return 0.7

        # Check for domain keywords in journal name
        keywords = self.get_keywords(domain)[:5]  # Use top domain keywords
        if any(keyword in journal_lower for keyword in keywords):
            return 0.5

        return 0.3

    def create_training_examples(self, domain: str = None) -> Dict[str, List[Dict]]:
        """
        Create training examples for all data types.

        Args:
            domain: Domain name (uses current domain if None)

        Returns:
            Dictionary with training examples by type
        """
        domain = domain or self.current_domain

        return {
            'book_data': self.get_sample_data('book', domain),
            'paper_data': self.get_sample_data('paper', domain),
            'integrated_data': self.get_sample_data('integrated', domain)
        }

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.

        Returns:
            Dictionary with configuration summary
        """
        domain_info = self.get_domain_info()

        return {
            'current_domain': self.current_domain,
            'domain_name': domain_info['name'],
            'domain_description': domain_info['description'],
            'available_domains': self.get_available_domains(),
            'journal_count': len(self.get_journals()),
            'keyword_count': len(self.get_keywords()),
            'foundational_author_count': len(self.get_foundational_authors()),
            'context_types': list(self.get_contexts().keys()),
            'pattern_types': list(self.get_patterns().keys()),
            'section_types': list(self.get_sections().keys()),
            'subdomain_types': list(self.get_subdomain_keywords().keys())
        }

    def __repr__(self) -> str:
        """String representation of domain configuration."""
        return f"DomainConfiguration(domain='{self.current_domain}', domains={len(self.get_available_domains())})"


# Global configuration instance (can be imported by other modules)
domain_config = None

def get_domain_config(config_path: str = None, domain: str = None) -> DomainConfiguration:
    """
    Get global domain configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call)
        domain: Domain to set (only used on first call)

    Returns:
        Global domain configuration instance
    """
    global domain_config

    if domain_config is None:
        domain_config = DomainConfiguration(config_path, domain)

    return domain_config

def set_global_domain(domain: str):
    """
    Set domain for global configuration instance.

    Args:
        domain: Domain name to set
    """
    global domain_config

    if domain_config is None:
        domain_config = DomainConfiguration(default_domain=domain)
    else:
        domain_config.set_domain(domain)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize domain configuration
        config = DomainConfiguration()

        print(f"Domain Configuration: {config}")
        print(f"Current domain: {config.current_domain}")
        print(f"Available domains: {config.get_available_domains()}")

        # Test domain switching
        print("\n" + "="*50)
        print("TESTING DOMAIN SWITCHING")
        print("="*50)

        for domain in config.get_available_domains():
            config.set_domain(domain)
            info = config.get_domain_info()
            keywords = config.get_keywords()

            print(f"\nDomain: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Keywords ({len(keywords)}): {keywords[:5]}...")
            print(f"Journals: {len(config.get_journals())}")
            print(f"Foundational Authors: {len(config.get_foundational_authors())}")

        # Test classification
        print("\n" + "="*50)
        print("TESTING TEXT CLASSIFICATION")
        print("="*50)

        test_texts = [
            "The benzene molecule has a hexagonal structure with resonance.",
            "Newton's laws of motion describe the relationship between forces and acceleration.",
            "The Pythagorean theorem states that a² + b² = c² in right triangles.",
            "DNA carries genetic information in all living organisms.",
            "The bridge design must account for wind loads and seismic activity."
        ]

        config.set_domain('chemistry')

        for text in test_texts:
            context = config.classify_context(text)
            relevance = config.validate_domain_relevance(text)
            print(f"Text: {text[:50]}...")
            print(f"Context: {context}, Relevance: {relevance:.2f}")
            print()

        # Configuration summary
        print("="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        summary = config.get_config_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")

    except DomainConfigurationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")