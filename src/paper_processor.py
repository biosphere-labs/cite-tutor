"""
Paper processing module that treats retrieved papers as foundational academic knowledge.
Focuses on pre-1960s papers as original sources of academic discoveries.
Supports multiple academic domains through configuration.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import gc
from dataclasses import dataclass

from pdf_processor import PDFExtractor
from structure_detector import MemoryManager
from domain_config import get_domain_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PaperKnowledge:
    """Data class for extracted paper knowledge."""
    knowledge_type: str
    content: str
    source_section: str
    historical_significance: str
    academic_domain: str
    confidence_score: float


@dataclass
class QAPair:
    """Data class for Q&A pairs generated from papers."""
    question: str
    answer: str
    knowledge_type: str
    source_type: str
    historical_context: str
    academic_domain: str
    confidence_score: float
    sources: List[str]


class CoreKnowledgeProcessor:
    """
    Processes retrieved academic papers as foundational knowledge sources.
    Treats pre-1960s papers as original discoveries, not supporting evidence.
    Supports multiple academic domains through configuration.
    """

    def __init__(self, config_path: str = "config/models.yaml", domain: str = None):
        self.config = self._load_config(config_path)
        self.memory_manager = MemoryManager(
            safety_buffer_mb=self.config.get("memory_limits", {}).get("safety_buffer_mb", 512)
        )
        self.pdf_extractor = PDFExtractor()

        # Initialize model variables (loaded on demand)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load domain configuration
        self.domain_config = get_domain_config()
        if domain:
            self.domain_config.set_domain(domain)

        logger.info(f"Core knowledge processor initialized for domain: {self.domain_config.current_domain}")

        # Historical significance keywords
        self.historical_keywords = {
            'first': ['first', 'initial', 'original', 'novel', 'new'],
            'discovery': ['discover', 'found', 'identify', 'isolate', 'synthesis'],
            'method': ['method', 'technique', 'procedure', 'approach', 'process'],
            'theory': ['theory', 'principle', 'concept', 'model', 'framework']
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration for 4GB VRAM."""
        return {
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

    def _load_model(self):
        """Load FLAN-T5-small model with memory optimization."""
        if self.model is not None:
            return

        model_name = self.config['qa_generation']['model']
        logger.info(f"Loading model: {model_name}")

        self.memory_manager.clear_memory()
        self.memory_manager.log_memory_usage("before model loading")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()

            self.memory_manager.log_memory_usage("after model loading")
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.memory_manager.clear_memory()
        logger.info("Model unloaded to free memory")

    def analyze_academic_paper_structure(self, paper_text: str) -> Dict:
        """Structure analysis specialized for academic papers."""

        # Split paper into paragraphs for analysis
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', paper_text) if p.strip()]

        # Get section patterns from domain configuration
        section_patterns = self.domain_config.get_sections()
        sections = {name: '' for name in section_patterns.keys()}
        sections['references'] = ''  # Always include references

        # Classify paragraphs into sections using pattern matching
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()

            # Determine section based on keywords and position
            classified = False
            for section_name in section_patterns.keys():
                try:
                    pattern = self.domain_config.get_compiled_pattern(section_name)
                    if pattern.search(paragraph_lower) and not classified:
                        sections[section_name] += paragraph + '\n\n'
                        classified = True
                        break
                except:
                    continue

            # If no specific section found, add to introduction (common for older papers)
            if not classified:
                intro_key = 'introduction' if 'introduction' in sections else list(sections.keys())[0]
                sections[intro_key] += paragraph + '\n\n'

        # Extract domain-specific content
        domain_content = {
            'entities_mentioned': self._extract_domain_entities(paper_text),
            'processes_described': self._extract_processes(paper_text),
            'mechanisms_proposed': self._extract_mechanisms(paper_text),
            'procedures': self._extract_procedures(paper_text),
            'theoretical_frameworks': self._extract_theories(paper_text),
            'historical_context': self._extract_historical_context(paper_text)
        }

        return {**sections, 'domain_content': domain_content}

    def _extract_domain_entities(self, text: str) -> List[str]:
        """Extract domain-specific entities mentioned in the paper."""
        entities = set()

        # Try to get domain-specific patterns
        try:
            # Look for compound patterns if available (chemistry domain)
            compounds_pattern = self.domain_config.get_compiled_pattern('compounds')
            matches = compounds_pattern.findall(text)
            entities.update(matches)
        except:
            # If no specific patterns, extract capitalized terms
            import re
            capitalized_terms = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities.update(capitalized_terms[:20])

        # Filter out common false positives
        false_positives = {'The', 'And', 'For', 'All', 'One', 'Two', 'New', 'Old', 'It', 'In', 'This', 'That'}
        entities = entities - false_positives

        return list(entities)[:20]  # Limit to top 20

    def _extract_processes(self, text: str) -> List[str]:
        """Extract process descriptions from the paper."""
        processes = []

        # Find sentences containing process indicators
        sentences = re.split(r'[.!?]+', text)
        try:
            # Try domain-specific process patterns
            reactions_pattern = self.domain_config.get_compiled_pattern('reactions')
            for sentence in sentences:
                if reactions_pattern.search(sentence):
                    cleaned = sentence.strip()
                    if len(cleaned) > 20 and len(cleaned) < 300:  # Reasonable length
                        processes.append(cleaned)
        except:
            # Fallback to generic process indicators
            process_keywords = ['process', 'method', 'procedure', 'technique', 'approach']
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in process_keywords):
                    cleaned = sentence.strip()
                    if len(cleaned) > 20 and len(cleaned) < 300:
                        processes.append(cleaned)

        return processes[:10]  # Limit to top 10

    def _extract_mechanisms(self, text: str) -> List[str]:
        """Extract mechanism discussions from the paper."""
        mechanisms = []

        # Look for mechanism-specific language
        mechanism_pattern = re.compile(r'\b(?:mechanism|pathway|intermediate|transition state|rate-determining)\b', re.IGNORECASE)
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            if mechanism_pattern.search(sentence):
                cleaned = sentence.strip()
                if len(cleaned) > 30:
                    mechanisms.append(cleaned)

        return mechanisms[:5]  # Limit to top 5

    def _extract_procedures(self, text: str) -> List[str]:
        """Extract experimental procedures from the paper."""
        procedures = []

        # Find paragraphs with procedural language
        paragraphs = re.split(r'\n\s*\n', text)
        try:
            # Try domain-specific procedure patterns
            procedures_pattern = self.domain_config.get_compiled_pattern('procedures')
            conditions_pattern = self.domain_config.get_compiled_pattern('conditions')

            for paragraph in paragraphs:
                if procedures_pattern.search(paragraph):
                    # Check if it contains experimental conditions
                    if conditions_pattern.search(paragraph):
                        procedures.append(paragraph.strip())
        except:
            # Fallback to generic procedure indicators
            procedure_keywords = ['procedure', 'method', 'experimental', 'analysis', 'measurement']
            for paragraph in paragraphs:
                if any(keyword in paragraph.lower() for keyword in procedure_keywords):
                    procedures.append(paragraph.strip())

        return procedures[:5]  # Limit to top 5

    def _extract_theories(self, text: str) -> List[str]:
        """Extract theoretical frameworks from the paper."""
        theories = []

        # Find theoretical discussions
        paragraphs = re.split(r'\n\s*\n', text)
        try:
            # Try domain-specific theory patterns
            theories_pattern = self.domain_config.get_compiled_pattern('theories')

            for paragraph in paragraphs:
                if theories_pattern.search(paragraph):
                    # Check for substantial theoretical content
                    if len(paragraph) > 100:
                        theories.append(paragraph.strip())
        except:
            # Fallback to generic theory indicators
            theory_keywords = ['theory', 'theoretical', 'model', 'framework', 'principle']
            for paragraph in paragraphs:
                if any(keyword in paragraph.lower() for keyword in theory_keywords):
                    if len(paragraph) > 100:
                        theories.append(paragraph.strip())

        return theories[:3]  # Limit to top 3

    def _extract_historical_context(self, text: str) -> Dict:
        """Extract historical significance of the work."""
        historical_context = {
            'novelty_claims': [],
            'previous_work_citations': [],
            'methodological_advances': []
        }

        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Look for novelty claims
            for keyword in self.historical_keywords['first']:
                if keyword in sentence_lower and ('time' in sentence_lower or 'report' in sentence_lower):
                    historical_context['novelty_claims'].append(sentence.strip())
                    break

            # Look for methodological advances
            for keyword in self.historical_keywords['method']:
                if keyword in sentence_lower and ('new' in sentence_lower or 'novel' in sentence_lower):
                    historical_context['methodological_advances'].append(sentence.strip())
                    break

        return historical_context

    def generate_foundational_qa_pairs(self, paper_structure: Dict, citation_context: Dict) -> List[QAPair]:
        """Generate Q&A focusing on foundational chemistry knowledge."""

        # Load model for Q&A generation
        self._load_model()

        qa_pairs = []

        try:
            # Original methodology questions
            if paper_structure['experimental']:
                methodology_qa = self._create_methodology_questions(
                    paper_structure['experimental'],
                    paper_structure['chemistry_content']['experimental_procedures'],
                    citation_context
                )
                qa_pairs.extend(methodology_qa)

            # Theoretical foundation questions
            if paper_structure['chemistry_content']['theoretical_frameworks']:
                theory_qa = self._create_theory_questions(
                    paper_structure['chemistry_content']['theoretical_frameworks'],
                    citation_context
                )
                qa_pairs.extend(theory_qa)

            # Historical significance questions
            historical_qa = self._create_historical_significance_questions(
                paper_structure, citation_context
            )
            qa_pairs.extend(historical_qa)

            # Discovery and innovation questions
            discovery_qa = self._create_discovery_questions(
                paper_structure, citation_context
            )
            qa_pairs.extend(discovery_qa)

        finally:
            # Always unload model to free memory
            self._unload_model()

        logger.info(f"Generated {len(qa_pairs)} foundational Q&A pairs")
        return qa_pairs

    def _create_methodology_questions(self, experimental_text: str, procedures: List[str], citation_context: Dict) -> List[QAPair]:
        """Create questions about original experimental methods."""
        qa_pairs = []

        for procedure in procedures:
            if len(procedure) < 50:  # Skip very short procedures
                continue

            # Generate methodology questions using AI
            questions_prompt = f"""
            Based on this historical chemistry experimental procedure, generate questions about the original methodology:

            Procedure: {procedure[:500]}

            Generate 2-3 questions about:
            - How the original method was performed
            - What conditions and reagents were used
            - What made this method innovative for its time

            Format: Question 1: [question] | Question 2: [question] | Question 3: [question]
            """

            try:
                # Generate questions with AI
                self.memory_manager.clear_memory()

                inputs = self.tokenizer.encode(questions_prompt, return_tensors="pt", max_length=400, truncation=True).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=150,
                        do_sample=False,
                        num_beams=1,
                        early_stopping=True
                    )

                questions_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                questions = [q.strip() for q in questions_text.split('|') if q.strip()]

                # Generate answers for each question
                for question in questions[:3]:  # Limit to 3 questions
                    answer = self._generate_methodological_answer(question, procedure, experimental_text)

                    qa_pairs.append(QAPair(
                        question=question,
                        answer=answer,
                        knowledge_type='foundational_methodology',
                        source_type='original_paper',
                        historical_context=self._extract_time_period_context(citation_context),
                        academic_domain=self._classify_academic_subdomain(procedure),
                        confidence_score=0.8,
                        sources=['original_experimental_section']
                    ))

                self.memory_manager.clear_memory()

            except Exception as e:
                logger.warning(f"Failed to generate methodology questions: {e}")
                continue

        return qa_pairs

    def _generate_methodological_answer(self, question: str, procedure: str, experimental_text: str) -> str:
        """Generate detailed methodological answer."""

        answer_prompt = f"""
        Answer this question about historical chemistry methodology:

        Question: {question}

        Experimental procedure: {procedure}

        Additional context: {experimental_text[:300]}

        Provide a detailed answer explaining the original method and its historical significance.
        """

        try:
            inputs = self.tokenizer.encode(answer_prompt, return_tensors="pt", max_length=400, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    do_sample=False,
                    num_beams=1
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer

        except Exception as e:
            logger.warning(f"Failed to generate answer: {e}")
            return f"Based on the original procedure: {procedure[:200]}..."

    def _create_theory_questions(self, theoretical_frameworks: List[str], citation_context: Dict) -> List[QAPair]:
        """Create questions about theoretical foundations."""
        qa_pairs = []

        for framework in theoretical_frameworks:
            # Focus on foundational theoretical contributions
            theory_prompt = f"""
            This is theoretical content from a foundational chemistry paper:

            {framework[:400]}

            Generate questions about:
            - What theoretical principle was proposed
            - How it changed understanding at the time
            - What evidence supported the theory

            Question 1: | Question 2: | Question 3:
            """

            try:
                inputs = self.tokenizer.encode(theory_prompt, return_tensors="pt", max_length=400, truncation=True).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(inputs, max_length=120, do_sample=False, num_beams=1)

                questions_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                questions = [q.strip() for q in questions_text.split('|') if q.strip()]

                for question in questions[:2]:  # Limit to 2 questions per framework
                    answer = f"According to this foundational work: {framework[:200]}. This theoretical contribution was significant because it {self._assess_theoretical_significance(framework)}."

                    qa_pairs.append(QAPair(
                        question=question,
                        answer=answer,
                        knowledge_type='foundational_theory',
                        source_type='original_paper',
                        historical_context=self._extract_time_period_context(citation_context),
                        academic_domain=self._classify_academic_subdomain(framework),
                        confidence_score=0.85,
                        sources=['theoretical_section']
                    ))

                self.memory_manager.clear_memory()

            except Exception as e:
                logger.warning(f"Failed to generate theory questions: {e}")
                continue

        return qa_pairs

    def _create_historical_significance_questions(self, paper_structure: Dict, citation_context: Dict) -> List[QAPair]:
        """Create questions about historical significance of the work."""
        qa_pairs = []

        historical_context = paper_structure['chemistry_content']['historical_context']

        # Questions about novelty claims
        if historical_context['novelty_claims']:
            for claim in historical_context['novelty_claims'][:2]:
                question = f"What was novel about this work when it was published?"
                answer = f"This work was significant because: {claim}. At the time, this represented {self._assess_historical_impact(claim, citation_context)}."

                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    knowledge_type='historical_significance',
                    source_type='original_paper',
                    historical_context=self._extract_time_period_context(citation_context),
                    chemistry_domain='general',
                    confidence_score=0.75,
                    sources=['full_paper']
                ))

        # Questions about methodological advances
        if historical_context['methodological_advances']:
            for advance in historical_context['methodological_advances'][:2]:
                question = f"What methodological advance did this paper introduce?"
                answer = f"The methodological contribution was: {advance}. This advanced the field by {self._assess_methodological_impact(advance)}."

                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    knowledge_type='methodological_innovation',
                    source_type='original_paper',
                    historical_context=self._extract_time_period_context(citation_context),
                    academic_domain=self._classify_academic_subdomain(advance),
                    confidence_score=0.8,
                    sources=['methodology_section']
                ))

        return qa_pairs

    def _create_discovery_questions(self, paper_structure: Dict, citation_context: Dict) -> List[QAPair]:
        """Create questions about discoveries and innovations."""
        qa_pairs = []

        entities = paper_structure['domain_content']['entities_mentioned']
        processes = paper_structure['domain_content']['processes_described']

        # Discovery questions about domain entities
        if entities:
            entity_type = 'entities' if self.domain_config.current_domain != 'chemistry' else 'compounds'
            question = f"What {entity_type} were discovered or studied in this work?"
            answer = f"This foundational work involved the following {entity_type}: {', '.join(entities[:5])}. These were significant because {self._assess_entity_significance(entities, citation_context)}."

            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                knowledge_type='domain_discovery',
                source_type='original_paper',
                historical_context=self._extract_time_period_context(citation_context),
                academic_domain=self._classify_academic_subdomain(' '.join(entities)),
                confidence_score=0.9,
                sources=['experimental_results']
            ))

        # Process discovery questions
        if processes:
            process_type = 'processes' if self.domain_config.current_domain != 'chemistry' else 'reactions'
            question = f"What {process_type} were developed in this work?"
            answer = f"The key {process_type} described include: {processes[0][:200]}... This represented a major advance in methodology for the time period."

            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                knowledge_type='process_discovery',
                source_type='original_paper',
                historical_context=self._extract_time_period_context(citation_context),
                academic_domain=self._classify_academic_subdomain(processes[0] if processes else ''),
                confidence_score=0.85,
                sources=['process_descriptions']
            ))

        return qa_pairs

    def _classify_academic_subdomain(self, text: str) -> str:
        """Classify academic subdomain based on text content."""
        return self.domain_config.classify_subdomain(text)

    def _extract_time_period_context(self, citation_context: Dict) -> str:
        """Extract time period context for historical significance."""
        year = citation_context.get('year', 'unknown')

        if isinstance(year, int):
            if year < 1900:
                return f"late 19th century ({year})"
            elif year < 1920:
                return f"early 20th century ({year})"
            elif year < 1950:
                return f"mid-20th century ({year})"
            else:
                return f"modern era ({year})"

        return "historical period"

    def _assess_theoretical_significance(self, framework: str) -> str:
        """Assess the significance of a theoretical contribution."""
        framework_lower = framework.lower()

        if 'first' in framework_lower or 'novel' in framework_lower:
            return "introduced a completely new theoretical framework"
        elif 'mechanism' in framework_lower:
            return "provided new mechanistic understanding"
        elif 'model' in framework_lower:
            return "established a new conceptual model"
        else:
            return "contributed to theoretical understanding"

    def _assess_historical_impact(self, claim: str, context: Dict) -> str:
        """Assess historical impact of a discovery claim."""
        claim_lower = claim.lower()

        if 'first' in claim_lower:
            return "the first demonstration of this phenomenon"
        elif 'new' in claim_lower:
            return "a novel contribution to the field"
        else:
            return "an important advance in chemical knowledge"

    def _assess_methodological_impact(self, advance: str) -> str:
        """Assess impact of methodological advances."""
        advance_lower = advance.lower()

        if 'method' in advance_lower:
            return "providing new experimental approaches"
        elif 'technique' in advance_lower:
            return "introducing new analytical techniques"
        else:
            return "advancing experimental methodology"

    def _assess_entity_significance(self, entities: List[str], context: Dict) -> str:
        """Assess significance of discovered entities."""
        if len(entities) > 5:
            return f"they represented a new class of {len(entities)} related entities"
        elif any(len(e) > 8 for e in entities):
            return f"they were complex {self.domain_config.current_domain} entities that were difficult to study at the time"
        else:
            return f"they provided new examples of important {self.domain_config.current_domain} structures"

    def process_retrieved_paper(self, pdf_bytes: bytes, citation_context: Dict) -> Dict:
        """
        Main method to process a retrieved paper as foundational knowledge.
        """
        logger.info(f"Processing retrieved paper from {citation_context.get('year', 'unknown')}")

        try:
            # Extract text from PDF
            paper_text = self.pdf_extractor.extract_text_from_bytes(pdf_bytes)
            if not paper_text:
                raise Exception("Failed to extract text from PDF")

            # Analyze paper structure
            paper_structure = self.analyze_academic_paper_structure(paper_text)

            # Generate foundational Q&A pairs
            qa_pairs = self.generate_foundational_qa_pairs(paper_structure, citation_context)

            # Assess paper quality and relevance
            quality_score = self._assess_paper_quality(paper_structure, citation_context)

            # Compile results
            processed_paper = {
                'citation_context': citation_context,
                'paper_structure': paper_structure,
                'qa_pairs': [
                    {
                        'question': qa.question,
                        'answer': qa.answer,
                        'knowledge_type': qa.knowledge_type,
                        'source_type': qa.source_type,
                        'historical_context': qa.historical_context,
                        'chemistry_domain': qa.chemistry_domain,
                        'confidence_score': qa.confidence_score,
                        'sources': qa.sources
                    } for qa in qa_pairs
                ],
                'quality_assessment': {
                    'overall_score': quality_score,
                    'text_length': len(paper_text),
                    'sections_found': len([s for s in paper_structure.values() if isinstance(s, str) and s.strip()]),
                    'domain_content_richness': len(paper_structure['domain_content']['entities_mentioned']) +
                                                 len(paper_structure['domain_content']['processes_described']),
                    'qa_pairs_generated': len(qa_pairs)
                },
                'processing_metadata': {
                    'processed_date': datetime.now().isoformat(),
                    'text_extraction_method': 'pdf_processor',
                    'model_used': self.config['qa_generation']['model'],
                    'foundational_paper': citation_context.get('year', 2000) < 1970
                }
            }

            logger.info(f"Successfully processed paper: {len(qa_pairs)} Q&A pairs generated, quality score: {quality_score:.2f}")
            return processed_paper

        except Exception as e:
            logger.error(f"Failed to process retrieved paper: {e}")
            raise

    def _assess_paper_quality(self, paper_structure: Dict, citation_context: Dict) -> float:
        """Assess quality and relevance of the processed paper."""
        score = 0.0

        # Text content quality (0-0.3)
        total_text = sum(len(str(section)) for section in paper_structure.values() if isinstance(section, str))
        if total_text > 5000:
            score += 0.3
        elif total_text > 2000:
            score += 0.2
        elif total_text > 1000:
            score += 0.1

        # Domain content richness (0-0.3)
        domain_content = paper_structure['domain_content']
        content_richness = (
            len(domain_content['entities_mentioned']) * 0.05 +
            len(domain_content['processes_described']) * 0.08 +
            len(domain_content['mechanisms_proposed']) * 0.1 +
            len(domain_content['procedures']) * 0.05 +
            len(domain_content['theoretical_frameworks']) * 0.1
        )
        score += min(0.3, content_richness)

        # Historical significance (0-0.2)
        year = citation_context.get('year')
        if year and year < 1940:
            score += 0.2
        elif year and year < 1960:
            score += 0.15
        elif year and year < 1980:
            score += 0.1

        # Section completeness (0-0.2)
        sections_with_content = sum(1 for section in paper_structure.values()
                                   if isinstance(section, str) and len(section.strip()) > 50)
        score += min(0.2, sections_with_content * 0.03)

        return min(1.0, score)

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Helper method to extract text from PDF bytes."""
        # This would typically use the PDFExtractor
        # For now, return placeholder
        return "PDF text extraction would be implemented here"

    def integrate_with_book_knowledge(self, paper_qa: List[Dict], book_context: Dict) -> List[Dict]:
        """Integrate paper Q&A with book knowledge context."""
        integrated_qa = []

        for qa in paper_qa:
            # Create enhanced Q&A that connects paper to book citation
            enhanced_qa = {
                **qa,
                'book_context': book_context,
                'integration_type': 'foundational_source',
                'knowledge_chain': f"Book cites -> Original paper contains -> {qa['knowledge_type']}"
            }

            # Add verification question
            verification_qa = {
                'question': f"How does this original research support what the textbook claims?",
                'answer': f"The original {paper_qa[0].get('historical_context', '')} research shows: {qa['answer'][:200]}... This directly supports the textbook's discussion of {book_context.get('context', 'this topic')}.",
                'knowledge_type': 'citation_verification',
                'source_type': 'integrated',
                'historical_context': qa.get('historical_context', ''),
                'chemistry_domain': qa.get('chemistry_domain', 'general'),
                'confidence_score': 0.9,
                'sources': ['original_paper', 'textbook_citation']
            }

            integrated_qa.extend([enhanced_qa, verification_qa])

        logger.info(f"Integrated {len(paper_qa)} paper Q&A with book context, created {len(integrated_qa)} enhanced Q&A pairs")
        return integrated_qa


if __name__ == "__main__":
    # Example usage
    processor = CoreKnowledgeProcessor()

    # Example paper processing
    citation_context = {
        'citation_text': 'J. Am. Chem. Soc. 53, 1367 (1931)',
        'year': 1931,
        'authors': ['Pauling'],
        'context': 'The concept of resonance was first introduced by Pauling'
    }

    # This would typically process actual PDF bytes
    print(f"Core Knowledge Processor initialized")
    print(f"Configuration: {processor.config}")
    print(f"Ready to process foundational chemistry papers")