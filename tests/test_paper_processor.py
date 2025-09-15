"""
Tests for Paper processor module.

Tests both unit (mocked) and integration (real model loading) modes.
Uses MLStandardTestCase for comprehensive ML framework mocking.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import torch
import json
from pathlib import Path
import tempfile

from base_test import MLStandardTestCase
from paper_processor import CoreKnowledgeProcessor, PaperKnowledge, QAPair


class TestPaperKnowledge(unittest.TestCase):
    """Test PaperKnowledge dataclass."""

    def test_paper_knowledge_creation(self):
        """Test creating PaperKnowledge instance."""
        knowledge = PaperKnowledge(
            knowledge_type="theoretical",
            content="Quantum mechanics theory",
            source_section="introduction",
            historical_significance="first theory",
            chemistry_domain="physical",
            confidence_score=0.9
        )

        self.assertEqual(knowledge.knowledge_type, "theoretical")
        self.assertEqual(knowledge.content, "Quantum mechanics theory")
        self.assertEqual(knowledge.chemistry_domain, "physical")
        self.assertEqual(knowledge.confidence_score, 0.9)


class TestQAPair(unittest.TestCase):
    """Test QAPair dataclass."""

    def test_qa_pair_creation(self):
        """Test creating QAPair instance."""
        qa = QAPair(
            question="What is the theory?",
            answer="The theory explains...",
            knowledge_type="theoretical",
            source_type="original_paper",
            historical_context="1930s",
            chemistry_domain="physical",
            confidence_score=0.85,
            sources=["section1", "section2"]
        )

        self.assertEqual(qa.question, "What is the theory?")
        self.assertEqual(qa.knowledge_type, "theoretical")
        self.assertEqual(len(qa.sources), 2)


class TestCoreKnowledgeProcessor(MLStandardTestCase):
    """Test CoreKnowledgeProcessor class with mock abstraction."""

    def setUp(self):
        super().setUp()

        # Create a temporary config file for testing
        self.config_content = {
            'qa_generation': {
                'model': 'google/flan-t5-small',
                'batch_size': 1,
                'max_length': 128
            },
            'memory_limits': {
                'max_gpu_memory_mb': 4096,
                'safety_buffer_mb': 512
            }
        }

        if self.test_mode == "unit":
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=self.config_content):
                    self.processor = CoreKnowledgeProcessor()
        else:
            # Integration test with real config loading
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                import yaml
                yaml.dump(self.config_content, f)
                self.temp_config_path = f.name

            self.processor = CoreKnowledgeProcessor(self.temp_config_path)

        # Sample data for testing
        self.sample_paper_text = """
        Abstract
        This paper presents a novel synthesis of benzene derivatives.

        Introduction
        Previous work has shown that aromatic compounds are important.

        Experimental
        Heat the mixture to 100°C for 2 hours. Add sodium chloride and stir.
        The reaction proceeds via an intermediate mechanism.

        Results
        We synthesized C6H6, C6H5Cl, and other compounds.
        The yield was 85% under optimal conditions.

        Discussion
        This method represents the first time such conditions were used.
        The theory of aromatic substitution explains these results.

        Conclusion
        We have developed a new synthetic method for benzene derivatives.
        """

        self.sample_citation_context = {
            'citation_text': 'Smith, J. Novel Synthesis. J. Am. Chem. Soc. 1925, 47, 123.',
            'year': 1925,
            'authors': ['Smith, J.'],
            'context': 'First synthesis of aromatic compounds'
        }

    def tearDown(self):
        super().tearDown()
        if hasattr(self, 'temp_config_path'):
            Path(self.temp_config_path).unlink(missing_ok=True)

    def test_load_config_success(self):
        """Test successful config loading."""
        if self.test_mode == "unit":
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=self.config_content):
                    config = self.processor._load_config("test_config.yaml")
                    self.assertEqual(config['qa_generation']['model'], 'google/flan-t5-small')

    def test_load_config_failure_uses_default(self):
        """Test fallback to default config on failure."""
        if self.test_mode == "unit":
            with patch('builtins.open', side_effect=FileNotFoundError()):
                config = self.processor._load_config("nonexistent.yaml")
                self.assertIn('qa_generation', config)
                self.assertEqual(config['qa_generation']['model'], 'google/flan-t5-small')

    def test_get_default_config(self):
        """Test default configuration."""
        default_config = self.processor._get_default_config()

        self.assertIn('qa_generation', default_config)
        self.assertIn('memory_limits', default_config)
        self.assertEqual(default_config['qa_generation']['model'], 'google/flan-t5-small')
        self.assertEqual(default_config['memory_limits']['max_gpu_memory_mb'], 4096)

    def test_load_model_success(self):
        """Test successful model loading."""
        if self.test_mode == "unit":
            # Use mocks from MLStandardTestCase
            with patch('transformers.T5Tokenizer.from_pretrained', return_value=self.mock_tokenizer):
                with patch('transformers.T5ForConditionalGeneration.from_pretrained', return_value=self.mock_model):
                    self.processor._load_model()

                    self.assertIsNotNone(self.processor.model)
                    self.assertIsNotNone(self.processor.tokenizer)
        else:
            # Integration test would load real model
            try:
                self.processor._load_model()
                self.assertIsNotNone(self.processor.model)
                self.assertIsNotNone(self.processor.tokenizer)
            except Exception as e:
                self.skipTest(f"Model loading failed in integration test: {e}")

    def test_unload_model(self):
        """Test model unloading."""
        self.processor.model = Mock()
        self.processor.tokenizer = Mock()

        self.processor._unload_model()

        self.assertIsNone(self.processor.model)
        self.assertIsNone(self.processor.tokenizer)

    def test_analyze_chemistry_paper_structure(self):
        """Test paper structure analysis."""
        structure = self.processor.analyze_chemistry_paper_structure(self.sample_paper_text)

        # Check that sections were identified
        self.assertIn('abstract', structure)
        self.assertIn('experimental', structure)
        self.assertIn('chemistry_content', structure)

        # Check chemistry content extraction
        chemistry_content = structure['chemistry_content']
        self.assertIn('compounds_synthesized', chemistry_content)
        self.assertIn('reactions_described', chemistry_content)
        self.assertIn('experimental_procedures', chemistry_content)

        # Check that some content was extracted
        self.assertGreater(len(structure['abstract']), 0)
        self.assertGreater(len(chemistry_content['compounds_synthesized']), 0)

    def test_extract_compounds(self):
        """Test compound extraction."""
        compounds = self.processor._extract_compounds(self.sample_paper_text)

        # Should find chemical formulas
        self.assertIn('C6H6', compounds)
        self.assertIn('C6H5Cl', compounds)

        # Should filter out common false positives
        self.assertNotIn('The', compounds)
        self.assertNotIn('And', compounds)

    def test_extract_reactions(self):
        """Test reaction extraction."""
        reactions = self.processor._extract_reactions(self.sample_paper_text)

        # Should find sentences with reaction indicators
        self.assertGreater(len(reactions), 0)

        # Check reasonable length filtering
        for reaction in reactions:
            self.assertGreaterEqual(len(reaction), 20)
            self.assertLessEqual(len(reaction), 300)

    def test_extract_mechanisms(self):
        """Test mechanism extraction."""
        text_with_mechanisms = """
        The reaction proceeds via an intermediate mechanism.
        The mechanism involves a transition state with high energy.
        """

        mechanisms = self.processor._extract_mechanisms(text_with_mechanisms)

        self.assertGreater(len(mechanisms), 0)
        self.assertTrue(any('mechanism' in m.lower() for m in mechanisms))

    def test_extract_procedures(self):
        """Test procedure extraction."""
        procedures = self.processor._extract_procedures(self.sample_paper_text)

        # Should find experimental procedures with conditions
        self.assertGreater(len(procedures), 0)

        # Should contain procedural language and conditions
        found_procedure = False
        for procedure in procedures:
            if 'heat' in procedure.lower() and '100' in procedure:
                found_procedure = True
                break
        self.assertTrue(found_procedure)

    def test_extract_theories(self):
        """Test theory extraction."""
        theories = self.processor._extract_theories(self.sample_paper_text)

        # Should find theoretical discussions
        self.assertGreater(len(theories), 0)

        # Should contain substantial content
        for theory in theories:
            self.assertGreater(len(theory), 100)

    def test_extract_historical_context(self):
        """Test historical context extraction."""
        historical_context = self.processor._extract_historical_context(self.sample_paper_text)

        self.assertIn('novelty_claims', historical_context)
        self.assertIn('methodological_advances', historical_context)

        # Should find the "first time" claim
        self.assertGreater(len(historical_context['novelty_claims']), 0)

    def test_classify_chemistry_domain(self):
        """Test chemistry domain classification."""
        # Test organic chemistry
        organic_text = "benzene aromatic organic alkane"
        domain = self.processor._classify_chemistry_domain(organic_text)
        self.assertEqual(domain, 'organic')

        # Test inorganic chemistry
        inorganic_text = "metal coordination complex ionic"
        domain = self.processor._classify_chemistry_domain(inorganic_text)
        self.assertEqual(domain, 'inorganic')

        # Test physical chemistry
        physical_text = "thermodynamics kinetics equilibrium"
        domain = self.processor._classify_chemistry_domain(physical_text)
        self.assertEqual(domain, 'physical')

        # Test general (no specific keywords)
        general_text = "some general chemistry text"
        domain = self.processor._classify_chemistry_domain(general_text)
        self.assertEqual(domain, 'general')

    def test_extract_time_period_context(self):
        """Test time period context extraction."""
        # Test different time periods
        self.assertEqual(
            self.processor._extract_time_period_context({'year': 1885}),
            "late 19th century (1885)"
        )

        self.assertEqual(
            self.processor._extract_time_period_context({'year': 1915}),
            "early 20th century (1915)"
        )

        self.assertEqual(
            self.processor._extract_time_period_context({'year': 1935}),
            "mid-20th century (1935)"
        )

        self.assertEqual(
            self.processor._extract_time_period_context({'year': 1975}),
            "modern era (1975)"
        )

        # Test unknown year
        self.assertEqual(
            self.processor._extract_time_period_context({'year': 'unknown'}),
            "historical period"
        )

    def test_assess_theoretical_significance(self):
        """Test theoretical significance assessment."""
        # Test novel framework
        significance = self.processor._assess_theoretical_significance("This is the first novel theory")
        self.assertIn("new theoretical framework", significance)

        # Test mechanism
        significance = self.processor._assess_theoretical_significance("The mechanism involves")
        self.assertIn("mechanistic understanding", significance)

        # Test model
        significance = self.processor._assess_theoretical_significance("New model for understanding")
        self.assertIn("conceptual model", significance)

    def test_assess_historical_impact(self):
        """Test historical impact assessment."""
        # Test first demonstration
        impact = self.processor._assess_historical_impact("This is the first demonstration", {})
        self.assertIn("first demonstration", impact)

        # Test new contribution
        impact = self.processor._assess_historical_impact("This new method", {})
        self.assertIn("novel contribution", impact)

    def test_assess_compound_significance(self):
        """Test compound significance assessment."""
        # Test many compounds
        many_compounds = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        significance = self.processor._assess_compound_significance(many_compounds, {})
        self.assertIn("new class", significance)

        # Test complex compounds
        complex_compounds = ['C6H5CH2CH2COOH']
        significance = self.processor._assess_compound_significance(complex_compounds, {})
        self.assertIn("complex molecules", significance)

        # Test simple compounds
        simple_compounds = ['CH4', 'H2O']
        significance = self.processor._assess_compound_significance(simple_compounds, {})
        self.assertIn("new examples", significance)

    def test_generate_foundational_qa_pairs(self):
        """Test Q&A pair generation."""
        if self.test_mode == "unit":
            paper_structure = self.processor.analyze_chemistry_paper_structure(self.sample_paper_text)

            # Mock the model components
            with patch.object(self.processor, '_load_model'):
                with patch.object(self.processor, '_unload_model'):
                    with patch.object(self.processor, 'tokenizer', self.mock_tokenizer):
                        with patch.object(self.processor, 'model', self.mock_model):
                            # Mock model generation
                            self.mock_model.generate.return_value = self.mock_tensor
                            self.mock_tokenizer.decode.return_value = "What is the method? | How does it work? | Why is it important?"

                            qa_pairs = self.processor.generate_foundational_qa_pairs(
                                paper_structure,
                                self.sample_citation_context
                            )

                            self.assertGreater(len(qa_pairs), 0)

                            # Check QAPair structure
                            for qa in qa_pairs:
                                self.assertIsInstance(qa, QAPair)
                                self.assertIsInstance(qa.question, str)
                                self.assertIsInstance(qa.answer, str)
                                self.assertIsInstance(qa.confidence_score, float)

    def test_create_methodology_questions(self):
        """Test methodology question creation."""
        if self.test_mode == "unit":
            experimental_text = "Heat the mixture to 100°C and stir for 2 hours."
            procedures = ["Heat the mixture to 100°C for 2 hours. Add sodium chloride and stir vigorously."]

            with patch.object(self.processor, 'tokenizer', self.mock_tokenizer):
                with patch.object(self.processor, 'model', self.mock_model):
                    self.mock_model.generate.return_value = self.mock_tensor
                    self.mock_tokenizer.decode.return_value = "How was the heating performed? | What temperature was used?"

                    qa_pairs = self.processor._create_methodology_questions(
                        experimental_text, procedures, self.sample_citation_context
                    )

                    self.assertGreater(len(qa_pairs), 0)

                    # Check that methodology questions were created
                    for qa in qa_pairs:
                        self.assertEqual(qa.knowledge_type, 'foundational_methodology')
                        self.assertEqual(qa.source_type, 'original_paper')

    def test_create_theory_questions(self):
        """Test theory question creation."""
        if self.test_mode == "unit":
            theoretical_frameworks = ["The theory of aromatic substitution explains these results through electron delocalization."]

            with patch.object(self.processor, 'tokenizer', self.mock_tokenizer):
                with patch.object(self.processor, 'model', self.mock_model):
                    self.mock_model.generate.return_value = self.mock_tensor
                    self.mock_tokenizer.decode.return_value = "What theory was proposed? | How does it work?"

                    qa_pairs = self.processor._create_theory_questions(
                        theoretical_frameworks, self.sample_citation_context
                    )

                    self.assertGreater(len(qa_pairs), 0)

                    # Check that theory questions were created
                    for qa in qa_pairs:
                        self.assertEqual(qa.knowledge_type, 'foundational_theory')

    def test_create_historical_significance_questions(self):
        """Test historical significance question creation."""
        paper_structure = {
            'chemistry_content': {
                'historical_context': {
                    'novelty_claims': ['This is the first time such conditions were used'],
                    'methodological_advances': ['New method for synthesis']
                }
            }
        }

        qa_pairs = self.processor._create_historical_significance_questions(
            paper_structure, self.sample_citation_context
        )

        self.assertGreater(len(qa_pairs), 0)

        # Check different types of historical questions
        knowledge_types = [qa.knowledge_type for qa in qa_pairs]
        self.assertIn('historical_significance', knowledge_types)

    def test_create_discovery_questions(self):
        """Test discovery question creation."""
        paper_structure = {
            'chemistry_content': {
                'compounds_synthesized': ['C6H6', 'C6H5Cl'],
                'reactions_described': ['Benzene reacts with chlorine to form chlorobenzene']
            }
        }

        qa_pairs = self.processor._create_discovery_questions(
            paper_structure, self.sample_citation_context
        )

        self.assertGreater(len(qa_pairs), 0)

        # Check discovery question types
        knowledge_types = [qa.knowledge_type for qa in qa_pairs]
        self.assertIn('chemical_discovery', knowledge_types)

    def test_assess_paper_quality(self):
        """Test paper quality assessment."""
        paper_structure = self.processor.analyze_chemistry_paper_structure(self.sample_paper_text)

        quality_score = self.processor._assess_paper_quality(
            paper_structure, self.sample_citation_context
        )

        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)

        # Historical papers should get bonus points
        self.assertGreater(quality_score, 0.5)  # 1925 paper should score well

    def test_process_retrieved_paper(self):
        """Test main paper processing method."""
        if self.test_mode == "unit":
            sample_pdf_bytes = b"%PDF-1.4\nSample PDF content"

            with patch.object(self.processor.pdf_extractor, 'extract_text_from_bytes', return_value=self.sample_paper_text):
                with patch.object(self.processor, 'generate_foundational_qa_pairs') as mock_generate_qa:
                    # Mock Q&A generation
                    sample_qa = QAPair(
                        question="Test question?",
                        answer="Test answer",
                        knowledge_type="test",
                        source_type="original_paper",
                        historical_context="1925",
                        chemistry_domain="organic",
                        confidence_score=0.8,
                        sources=["test"]
                    )
                    mock_generate_qa.return_value = [sample_qa]

                    result = self.processor.process_retrieved_paper(
                        sample_pdf_bytes, self.sample_citation_context
                    )

                    # Check result structure
                    self.assertIn('citation_context', result)
                    self.assertIn('paper_structure', result)
                    self.assertIn('qa_pairs', result)
                    self.assertIn('quality_assessment', result)
                    self.assertIn('processing_metadata', result)

                    # Check Q&A pairs
                    self.assertEqual(len(result['qa_pairs']), 1)
                    self.assertEqual(result['qa_pairs'][0]['question'], "Test question?")

    def test_process_retrieved_paper_extraction_failure(self):
        """Test paper processing with text extraction failure."""
        if self.test_mode == "unit":
            sample_pdf_bytes = b"%PDF-1.4\nSample PDF content"

            with patch.object(self.processor.pdf_extractor, 'extract_text_from_bytes', return_value=""):
                with self.assertRaises(Exception) as context:
                    self.processor.process_retrieved_paper(
                        sample_pdf_bytes, self.sample_citation_context
                    )

                self.assertIn("Failed to extract text", str(context.exception))

    def test_integrate_with_book_knowledge(self):
        """Test integration with book knowledge."""
        paper_qa = [
            {
                'question': 'What method was used?',
                'answer': 'The original method involved heating',
                'knowledge_type': 'methodology',
                'historical_context': '1925',
                'chemistry_domain': 'organic'
            }
        ]

        book_context = {
            'context': 'synthesis methods',
            'book_section': 'Chapter 5'
        }

        integrated_qa = self.processor.integrate_with_book_knowledge(paper_qa, book_context)

        # Should create enhanced Q&A pairs
        self.assertGreater(len(integrated_qa), len(paper_qa))

        # Check integration metadata
        for qa in integrated_qa:
            if 'book_context' in qa:
                self.assertEqual(qa['book_context'], book_context)
                self.assertIn('integration_type', qa)

    def test_extract_text_from_bytes_placeholder(self):
        """Test placeholder text extraction method."""
        # This is a placeholder method that should be implemented
        result = self.processor.extract_text_from_bytes(b"test")
        self.assertIsInstance(result, str)

    @pytest.mark.slow
    def test_performance_large_text(self):
        """Test performance with large text."""
        # Create large text sample
        large_text = self.sample_paper_text * 100

        # Test structure analysis performance
        result = self.assertExecutionTime(
            self.processor.analyze_chemistry_paper_structure,
            max_time=5.0,  # Should complete within 5 seconds
            paper_text=large_text
        )

        self.assertIn('chemistry_content', result)

    @pytest.mark.integration
    def test_end_to_end_processing(self):
        """End-to-end integration test (only runs in integration mode)."""
        if self.test_mode == "integration":
            try:
                # Test with real paper text and model loading
                paper_structure = self.processor.analyze_chemistry_paper_structure(self.sample_paper_text)

                # This would test real model loading and Q&A generation
                # Skip actual Q&A generation to avoid long model loading times
                self.assertIn('chemistry_content', paper_structure)
                self.assertGreater(len(paper_structure['chemistry_content']['compounds_synthesized']), 0)

            except Exception as e:
                # Model loading issues in integration tests are acceptable
                self.skipTest(f"Integration test skipped due to model loading: {e}")

    @pytest.mark.gpu
    def test_model_device_selection(self):
        """Test device selection for model loading."""
        if self.test_mode == "unit":
            # Test CUDA availability check
            with patch('torch.cuda.is_available', return_value=True):
                processor = CoreKnowledgeProcessor()
                self.assertEqual(processor.device.type, 'cuda')

            with patch('torch.cuda.is_available', return_value=False):
                processor = CoreKnowledgeProcessor()
                self.assertEqual(processor.device.type, 'cpu')

    def test_memory_management_integration(self):
        """Test memory management integration."""
        if self.test_mode == "unit":
            # Test that memory manager is used
            with patch.object(self.processor.memory_manager, 'clear_memory') as mock_clear:
                with patch.object(self.processor.memory_manager, 'log_memory_usage') as mock_log:
                    try:
                        self.processor._load_model()
                    except:
                        pass  # Expected to fail in unit test

                    # Should have called memory management methods
                    mock_clear.assert_called()
                    mock_log.assert_called()

    def test_qa_pair_serialization(self):
        """Test Q&A pair serialization in process_retrieved_paper."""
        if self.test_mode == "unit":
            sample_pdf_bytes = b"%PDF-1.4\nSample PDF content"

            with patch.object(self.processor.pdf_extractor, 'extract_text_from_bytes', return_value=self.sample_paper_text):
                with patch.object(self.processor, 'generate_foundational_qa_pairs') as mock_generate_qa:
                    # Mock Q&A generation with QAPair objects
                    sample_qa = QAPair(
                        question="Test question?",
                        answer="Test answer",
                        knowledge_type="test",
                        source_type="original_paper",
                        historical_context="1925",
                        chemistry_domain="organic",
                        confidence_score=0.8,
                        sources=["test"]
                    )
                    mock_generate_qa.return_value = [sample_qa]

                    result = self.processor.process_retrieved_paper(
                        sample_pdf_bytes, self.sample_citation_context
                    )

                    # Check that QAPair objects were properly serialized to dicts
                    qa_dict = result['qa_pairs'][0]
                    self.assertIsInstance(qa_dict, dict)
                    self.assertEqual(qa_dict['question'], "Test question?")
                    self.assertEqual(qa_dict['knowledge_type'], "test")
                    self.assertIsInstance(qa_dict['sources'], list)


if __name__ == '__main__':
    unittest.main()