"""
Example script demonstrating how to use the domain configuration system.

This script shows how the sci-tutor system can now work with different academic domains
by simply changing the domain configuration, without any code changes.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from domain_config import get_domain_config, set_global_domain
from scholar_scraper import ScholarScraper
from citation_extractor import AcademicCitationExtractor
from paper_processor import CoreKnowledgeProcessor
from training_manager import UniversalMultiStageTrainer, create_sample_datasets
from enhanced_fine_tuner import IntegratedFineTuner


def demonstrate_domain_switching():
    """Demonstrate how the system works with different domains."""

    print("=" * 80)
    print("SCI-TUTOR DOMAIN CONFIGURATION SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Get domain configuration
    config = get_domain_config()

    print(f"Available domains: {config.get_available_domains()}")
    print()

    # Test each domain
    for domain in config.get_available_domains():
        print(f"\n{'='*60}")
        print(f"TESTING DOMAIN: {domain.upper()}")
        print(f"{'='*60}")

        # Switch to domain
        config.set_domain(domain)

        # Show domain information
        info = config.get_domain_info()
        print(f"Name: {info['name']}")
        print(f"Description: {info['description']}")

        # Show some configuration details
        keywords = config.get_keywords()
        journals = config.get_journals()
        contexts = config.get_contexts()

        print(f"\nConfiguration Details:")
        print(f"- Keywords: {len(keywords)} (first 5: {keywords[:5]})")
        print(f"- Journals: {len(journals)}")
        print(f"- Contexts: {list(contexts.keys())}")

        # Test text classification
        test_texts = {
            'chemistry': "The benzene molecule has a hexagonal structure with resonance.",
            'physics': "Newton's laws of motion describe the relationship between forces and acceleration.",
            'mathematics': "The Pythagorean theorem states that a² + b² = c² in right triangles.",
            'biology': "DNA carries genetic information in all living organisms.",
            'engineering': "The bridge design must account for wind loads and seismic activity."
        }

        test_text = test_texts.get(domain, "This is a sample academic text.")
        relevance = config.validate_domain_relevance(test_text)
        context = config.classify_context(test_text)

        print(f"\nText Classification for '{test_text[:50]}...':")
        print(f"- Domain relevance: {relevance:.2f}")
        print(f"- Classified context: {context}")

        # Show sample training data
        sample_data = config.get_sample_data('book')
        if sample_data:
            print(f"\nSample book data:")
            for i, sample in enumerate(sample_data[:2]):
                print(f"  Q{i+1}: {sample['question']}")
                print(f"  A{i+1}: {sample['answer']}")


def demonstrate_component_usage():
    """Demonstrate how different components use domain configuration."""

    print(f"\n{'='*60}")
    print("COMPONENT USAGE DEMONSTRATION")
    print(f"{'='*60}")

    # Set domain to chemistry for this example
    set_global_domain('chemistry')

    print("\n1. Scholar Scraper with domain configuration:")
    scraper = ScholarScraper(domain='chemistry')

    # Example journal abbreviation expansion
    test_citation = "J. Am. Chem. Soc. 85, 2544 (1963)"
    expanded = scraper.expand_journal_abbreviations(test_citation)
    print(f"   Original: {test_citation}")
    print(f"   Expanded: {expanded}")

    print("\n2. Citation Extractor with domain configuration:")
    extractor = AcademicCitationExtractor(domain='chemistry')

    # Example text with citations
    sample_text = """
    The concept of resonance was first introduced by Pauling (J. Am. Chem. Soc. 53, 1367 (1931))
    and later developed in his comprehensive work. The benzene structure involves
    delocalized electrons and aromatic stabilization.
    """

    citations = extractor.extract_citations_from_text(sample_text)
    print(f"   Extracted {len(citations)} citations")
    for citation in citations:
        print(f"   - Citation: {citation.citation_text}")
        print(f"     Context: {citation.academic_context}")
        print(f"     Entities: {citation.entities_mentioned}")
        print(f"     Priority: {citation.retrieval_priority}")

    print("\n3. Paper Processor with domain configuration:")
    processor = CoreKnowledgeProcessor(domain='chemistry')

    print(f"   Processor initialized for domain: {processor.domain_config.current_domain}")
    print(f"   Available contexts: {list(processor.domain_config.get_contexts().keys())}")

    print("\n4. Training Manager with sample datasets:")
    trainer = UniversalMultiStageTrainer(budget_limit=10.0)

    # Create domain-specific sample datasets
    book_data, paper_data, integrated_data = create_sample_datasets('chemistry')
    print(f"   Created sample datasets:")
    print(f"   - Book data: {len(book_data)} examples")
    print(f"   - Paper data: {len(paper_data)} examples")
    print(f"   - Integrated data: {len(integrated_data)} examples")

    # Show first example from each dataset
    print(f"   Book example: Q: {book_data[0]['question'][:50]}...")
    print(f"                 A: {book_data[0]['answer'][:50]}...")


def demonstrate_cross_domain_comparison():
    """Compare how the same text is processed across different domains."""

    print(f"\n{'='*60}")
    print("CROSS-DOMAIN COMPARISON")
    print(f"{'='*60}")

    # Test text that could relate to multiple domains
    test_text = "The quantum mechanical model describes electron behavior in atomic orbitals using wave functions and probability distributions."

    config = get_domain_config()

    print(f"Test text: {test_text}")
    print()

    for domain in ['chemistry', 'physics', 'mathematics']:
        config.set_domain(domain)

        relevance = config.validate_domain_relevance(test_text)
        context = config.classify_context(test_text)
        subdomain = config.classify_subdomain(test_text)

        print(f"{domain.capitalize():12} - Relevance: {relevance:.2f}, Context: {context:15}, Subdomain: {subdomain}")


def main():
    """Run all demonstrations."""
    try:
        demonstrate_domain_switching()
        demonstrate_component_usage()
        demonstrate_cross_domain_comparison()

        print(f"\n{'='*80}")
        print("DOMAIN CONFIGURATION SYSTEM DEMONSTRATION COMPLETE")
        print("="*80)
        print("The sci-tutor system now supports multiple academic domains through configuration!")
        print("You can switch domains by simply changing the domain configuration.")
        print("All hardcoded domain-specific patterns have been abstracted into the configuration file.")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()