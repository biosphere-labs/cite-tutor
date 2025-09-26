"""
AI-powered document structure analyzer optimized for 4GB VRAM and chemistry content.
Uses FLAN-T5-small for intelligent text analysis while maintaining memory constraints.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import psutil
import gc
from dataclasses import dataclass
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChunkAnalysis:
    """Data class for chunk analysis results."""
    chunk_index: int
    content_type: str
    chemistry_topic: str
    key_compounds: List[str]
    flow_type: str  # 'new' or 'continuation'
    confidence: float
    chunk_length: int


@dataclass
class DocumentSection:
    """Data class for document sections."""
    section_id: str
    title: str
    chunks: List[int]  # Chunk indices
    content_type: str
    chemistry_topic: str
    start_position: int
    end_position: int


class MemoryManager:
    """Memory management utilities for 4GB VRAM optimization."""

    def __init__(self, safety_buffer_mb: int = 512):
        self.safety_buffer_mb = safety_buffer_mb
        self.max_memory_mb = 4096 - safety_buffer_mb

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "total": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "cached": torch.cuda.memory_reserved() / 1024**2,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**2
        }

    def clear_memory(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def check_memory_available(self, required_mb: float = 1000) -> bool:
        """Check if enough memory is available for operation."""
        memory = self.get_gpu_memory_usage()
        available = memory["total"] - memory["allocated"]
        return available >= required_mb

    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        memory = self.get_gpu_memory_usage()
        logger.info(f"Memory {context}: Allocated: {memory['allocated']:.0f}MB, "
                   f"Cached: {memory['cached']:.0f}MB, "
                   f"Available: {memory['total'] - memory['allocated']:.0f}MB")


class DocumentStructureAnalyzer:
    """
    AI-powered document structure analyzer for chemistry books.
    Optimized for 4GB VRAM with intelligent chunking and semantic analysis.
    """

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config = self._load_config(config_path)
        self.memory_manager = MemoryManager(
            safety_buffer_mb=self.config.get("memory_limits", {}).get("safety_buffer_mb", 512)
        )

        # Initialize model variables (loaded on demand)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Chemistry-specific patterns
        self.chemistry_patterns = {
            'synthesis': re.compile(r'\b(?:synthesis|synthesize|preparation|prepared|reaction|yield|product)\b', re.IGNORECASE),
            'mechanism': re.compile(r'\b(?:mechanism|pathway|intermediate|transition|state|catalyst|catalytic)\b', re.IGNORECASE),
            'analysis': re.compile(r'\b(?:analysis|characterization|spectroscopy|nmr|ir|mass|spectr|chromatography)\b', re.IGNORECASE),
            'theory': re.compile(r'\b(?:theory|principle|concept|fundamental|basis|foundation|model)\b', re.IGNORECASE),
            'data': re.compile(r'\b(?:results?|data|table|figure|chart|graph|measurement|observation)\b', re.IGNORECASE),
            'literature': re.compile(r'\b(?:reference|citation|literature|study|research|reported|published)\b', re.IGNORECASE),
            'procedure': re.compile(r'\b(?:procedure|method|protocol|step|experimental|condition)\b', re.IGNORECASE),
            'equation': re.compile(r'(?:equation|formula|expression|\d+\.\d+|\w+\s*=\s*\w+|→|↔)', re.IGNORECASE),
        }

        # Chemistry topic patterns
        self.topic_patterns = {
            'organic': re.compile(r'\b(?:organic|alkane|alkene|aromatic|benzene|carbonyl|carboxylic|amino|polymer)\b', re.IGNORECASE),
            'inorganic': re.compile(r'\b(?:inorganic|metal|coordination|complex|crystal|ionic|salt|oxide)\b', re.IGNORECASE),
            'physical': re.compile(r'\b(?:thermodynamics|kinetics|equilibrium|phase|solution|colligative|quantum)\b', re.IGNORECASE),
            'analytical': re.compile(r'\b(?:analytical|quantitative|qualitative|titration|gravimetric|electrochemical)\b', re.IGNORECASE),
            'biochemistry': re.compile(r'\b(?:biochemistry|enzyme|protein|amino acid|dna|rna|metabolism)\b', re.IGNORECASE),
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
            'structure_analysis': {
                'model': 'google/flan-t5-small',
                'max_memory_mb': 1024
            },
            'memory_limits': {
                'max_gpu_memory_mb': 4096,
                'safety_buffer_mb': 512
            }
        }

    def _load_model(self):
        """Load FLAN-T5-small model with memory optimization."""
        if self.model is not None:
            return  # Already loaded

        model_name = self.config['structure_analysis']['model']
        logger.info(f"Loading model: {model_name}")

        self.memory_manager.clear_memory()
        self.memory_manager.log_memory_usage("before model loading")

        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)

            # Load model with memory optimization
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)

            # Set to evaluation mode
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

    def chunk_text_intelligently(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Chemistry-aware intelligent text chunking.
        Preserves chemical equations, reactions, and compound discussions.
        """
        # First, split by major boundaries (chapters, sections)
        major_boundaries = re.split(r'\n\s*(?:Chapter|Section|CHAPTER|SECTION|\d+\.\d*)\s*[^\n]*\n', text)

        chunks = []

        for section in major_boundaries:
            if not section.strip():
                continue

            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', section)

            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Check if paragraph contains chemical equations or reactions
                has_equation = bool(re.search(r'[→←↔⇌]|(?:\w+\s*=\s*\w+)|(?:\d+\.\d+)', para))
                has_mechanism = bool(self.chemistry_patterns['mechanism'].search(para))
                has_synthesis = bool(self.chemistry_patterns['synthesis'].search(para))

                # If adding this paragraph would exceed chunk size
                if len(current_chunk) + len(para) > max_chunk_size:
                    # Don't break if it's part of a chemical discussion
                    if (has_equation or has_mechanism or has_synthesis) and len(current_chunk) > 0:
                        # Try to find a natural break point
                        sentences = re.split(r'[.!?]+', current_chunk)
                        if len(sentences) > 2:
                            # Keep most sentences, start new chunk with last sentence + new para
                            keep_sentences = sentences[:-1]
                            current_chunk = '. '.join(keep_sentences) + '.'
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentences[-1] + '. ' + para
                        else:
                            # Keep the chunk together
                            current_chunk += '\n\n' + para
                    else:
                        # Safe to break here
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = para
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += '\n\n' + para
                    else:
                        current_chunk = para

            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        # Filter out very short chunks and merge them
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk) < 100 and filtered_chunks:
                # Merge with previous chunk if it won't exceed size limit
                if len(filtered_chunks[-1]) + len(chunk) < max_chunk_size * 1.2:
                    filtered_chunks[-1] += '\n\n' + chunk
                else:
                    filtered_chunks.append(chunk)
            else:
                filtered_chunks.append(chunk)

        logger.info(f"Text chunked into {len(filtered_chunks)} intelligent chunks")
        return filtered_chunks

    def analyze_chunk_semantic_role(self, chunk: str, chunk_index: int) -> ChunkAnalysis:
        """AI analysis optimized for chemistry content with memory management."""

        # Ensure model is loaded
        self._load_model()

        # Clear memory before processing
        self.memory_manager.clear_memory()

        # Create analysis prompt
        analysis_prompt = f"""Analyze this chemistry text chunk and identify its role:

Chunk {chunk_index}: {chunk[:600]}{'...' if len(chunk) > 600 else ''}

Chemistry content types:
- Introduction: Introduces chemical concepts
- Experimental: Experimental procedures/synthesis
- Mechanism: Reaction mechanisms discussion
- Results: Results and data analysis
- Theory: Theoretical background/principles
- Literature: Literature references/citations
- Math: Mathematical derivations

Chemistry topics: organic, inorganic, physical, analytical, biochemistry

Key compounds: List main chemical compounds mentioned

Flow type: new (introduces concepts) or continuation (continues discussion)

Format: Type: [type] | Topic: [topic] | Compounds: [compounds] | Flow: [flow]
"""

        try:
            # Tokenize with length limit
            inputs = self.tokenizer.encode(
                analysis_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            # Generate with memory constraints
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=100,
                    do_sample=False,
                    num_beams=1,  # Reduce memory usage
                    early_stopping=True
                )

            # Decode result
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clear memory after processing
            self.memory_manager.clear_memory()

            return self._parse_chemistry_analysis(result, chunk, chunk_index)

        except Exception as e:
            logger.error(f"AI analysis failed for chunk {chunk_index}: {e}")
            # Fallback to pattern-based analysis
            return self._fallback_pattern_analysis(chunk, chunk_index)

    def _parse_chemistry_analysis(self, ai_result: str, chunk: str, chunk_index: int) -> ChunkAnalysis:
        """Parse AI analysis result into structured format."""

        # Default values
        content_type = "introduction"
        chemistry_topic = "general"
        key_compounds = []
        flow_type = "new"
        confidence = 0.5

        try:
            # Parse AI result format: Type: [type] | Topic: [topic] | Compounds: [compounds] | Flow: [flow]
            parts = ai_result.lower().split('|')

            for part in parts:
                part = part.strip()
                if part.startswith('type:'):
                    content_type = part.split(':')[1].strip()
                elif part.startswith('topic:'):
                    chemistry_topic = part.split(':')[1].strip()
                elif part.startswith('compounds:'):
                    compounds_str = part.split(':', 1)[1].strip()
                    key_compounds = [c.strip() for c in compounds_str.split(',') if c.strip()]
                elif part.startswith('flow:'):
                    flow_type = part.split(':')[1].strip()

            confidence = 0.8  # High confidence for successful AI parsing

        except Exception as e:
            logger.warning(f"Failed to parse AI result for chunk {chunk_index}: {e}")
            # Use fallback pattern analysis
            return self._fallback_pattern_analysis(chunk, chunk_index)

        return ChunkAnalysis(
            chunk_index=chunk_index,
            content_type=content_type,
            chemistry_topic=chemistry_topic,
            key_compounds=key_compounds,
            flow_type=flow_type,
            confidence=confidence,
            chunk_length=len(chunk)
        )

    def _fallback_pattern_analysis(self, chunk: str, chunk_index: int) -> ChunkAnalysis:
        """Fallback pattern-based analysis when AI fails."""

        chunk_lower = chunk.lower()

        # Determine content type based on patterns
        content_type = "introduction"
        max_score = 0

        for pattern_name, pattern in self.chemistry_patterns.items():
            matches = len(pattern.findall(chunk))
            if matches > max_score:
                max_score = matches
                content_type = pattern_name

        # Determine chemistry topic
        chemistry_topic = "general"
        max_topic_score = 0

        for topic_name, pattern in self.topic_patterns.items():
            matches = len(pattern.findall(chunk))
            if matches > max_topic_score:
                max_topic_score = matches
                chemistry_topic = topic_name

        # Extract key compounds (simple pattern)
        compound_pattern = re.compile(r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*\b')
        potential_compounds = compound_pattern.findall(chunk)
        key_compounds = [c for c in potential_compounds[:5] if len(c) > 1]  # Top 5, filter single letters

        # Determine flow type (heuristic)
        flow_indicators = ['therefore', 'thus', 'consequently', 'furthermore', 'moreover', 'additionally']
        flow_type = "continuation" if any(indicator in chunk_lower for indicator in flow_indicators) else "new"

        return ChunkAnalysis(
            chunk_index=chunk_index,
            content_type=content_type,
            chemistry_topic=chemistry_topic,
            key_compounds=key_compounds,
            flow_type=flow_type,
            confidence=0.6,  # Medium confidence for pattern-based
            chunk_length=len(chunk)
        )

    def group_chunks_into_sections(self, chunks: List[str], analyses: List[ChunkAnalysis]) -> List[DocumentSection]:
        """
        Group related chemistry chunks into coherent sections.
        Maintains chemistry concept flow and reaction sequences.
        """

        sections = []
        current_section = None
        section_counter = 1

        for i, analysis in enumerate(analyses):
            # Start new section conditions:
            # 1. First chunk
            # 2. New topic introduction
            # 3. Major content type change
            # 4. Topic change

            should_start_new_section = (
                current_section is None or
                (analysis.flow_type == "new" and analysis.content_type in ["introduction", "theory"]) or
                (current_section and analysis.chemistry_topic != current_section.chemistry_topic and
                 analysis.content_type != "continuation") or
                (current_section and len(current_section.chunks) > 10)  # Prevent overly long sections
            )

            if should_start_new_section:
                # Finish previous section
                if current_section:
                    sections.append(current_section)

                # Create new section
                section_title = self._generate_section_title(analysis, section_counter)
                current_section = DocumentSection(
                    section_id=f"section_{section_counter:03d}",
                    title=section_title,
                    chunks=[i],
                    content_type=analysis.content_type,
                    chemistry_topic=analysis.chemistry_topic,
                    start_position=i,
                    end_position=i
                )
                section_counter += 1
            else:
                # Add to current section
                current_section.chunks.append(i)
                current_section.end_position = i

                # Update section type if more specific content found
                if analysis.content_type in ["experimental", "mechanism", "results"] and \
                   current_section.content_type == "introduction":
                    current_section.content_type = analysis.content_type

        # Add final section
        if current_section:
            sections.append(current_section)

        logger.info(f"Grouped {len(chunks)} chunks into {len(sections)} sections")
        return sections

    def _generate_section_title(self, analysis: ChunkAnalysis, section_number: int) -> str:
        """Generate descriptive section title based on analysis."""

        topic_titles = {
            'organic': 'Organic Chemistry',
            'inorganic': 'Inorganic Chemistry',
            'physical': 'Physical Chemistry',
            'analytical': 'Analytical Chemistry',
            'biochemistry': 'Biochemistry'
        }

        content_titles = {
            'introduction': 'Introduction',
            'experimental': 'Experimental Methods',
            'mechanism': 'Reaction Mechanisms',
            'results': 'Results and Analysis',
            'theory': 'Theoretical Background',
            'literature': 'Literature Review',
            'procedure': 'Procedures'
        }

        topic_title = topic_titles.get(analysis.chemistry_topic, 'Chemistry')
        content_title = content_titles.get(analysis.content_type, 'Discussion')

        if analysis.key_compounds:
            compounds_str = ', '.join(analysis.key_compounds[:3])  # Top 3 compounds
            return f"{section_number}. {topic_title}: {content_title} ({compounds_str})"
        else:
            return f"{section_number}. {topic_title}: {content_title}"

    def analyze_document_structure(self, text: str, book_name: str = "Chemistry Book") -> Dict:
        """
        Main method to analyze document structure with chemistry awareness.
        Returns comprehensive structure analysis optimized for 4GB VRAM.
        """

        logger.info(f"Starting document structure analysis for: {book_name}")
        self.memory_manager.log_memory_usage("start")

        try:
            # Step 1: Intelligent chunking
            logger.info("Performing chemistry-aware text chunking...")
            chunks = self.chunk_text_intelligently(text)

            # Step 2: Analyze each chunk semantically
            logger.info(f"Analyzing {len(chunks)} chunks with AI...")
            analyses = []

            # Process chunks sequentially to manage memory
            for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
                try:
                    analysis = self.analyze_chunk_semantic_role(chunk, i)
                    analyses.append(analysis)

                    # Memory check every 10 chunks
                    if (i + 1) % 10 == 0:
                        self.memory_manager.log_memory_usage(f"after chunk {i+1}")

                        # Force memory cleanup if getting full
                        memory = self.memory_manager.get_gpu_memory_usage()
                        if memory["allocated"] > self.memory_manager.max_memory_mb * 0.8:
                            logger.warning("Memory usage high, forcing cleanup...")
                            self._unload_model()
                            self.memory_manager.clear_memory()

                except Exception as e:
                    logger.error(f"Failed to analyze chunk {i}: {e}")
                    # Create fallback analysis
                    analyses.append(self._fallback_pattern_analysis(chunk, i))

            # Step 3: Group chunks into sections
            logger.info("Grouping chunks into coherent sections...")
            sections = self.group_chunks_into_sections(chunks, analyses)

            # Step 4: Compile results
            result = {
                'book_name': book_name,
                'total_chunks': len(chunks),
                'total_sections': len(sections),
                'chunks': [
                    {
                        'index': i,
                        'text': chunk,
                        'length': len(chunk),
                        'analysis': {
                            'content_type': analyses[i].content_type,
                            'chemistry_topic': analyses[i].chemistry_topic,
                            'key_compounds': analyses[i].key_compounds,
                            'flow_type': analyses[i].flow_type,
                            'confidence': analyses[i].confidence
                        }
                    } for i, chunk in enumerate(chunks)
                ],
                'sections': [
                    {
                        'section_id': section.section_id,
                        'title': section.title,
                        'chunks': section.chunks,
                        'content_type': section.content_type,
                        'chemistry_topic': section.chemistry_topic,
                        'start_position': section.start_position,
                        'end_position': section.end_position,
                        'text_length': sum(len(chunks[idx]) for idx in section.chunks)
                    } for section in sections
                ],
                'summary': self._generate_document_summary(analyses, sections)
            }

            # Clean up
            self._unload_model()
            self.memory_manager.log_memory_usage("end")

            logger.info(f"Document structure analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"Document structure analysis failed: {e}")
            self._unload_model()
            raise

    def _generate_document_summary(self, analyses: List[ChunkAnalysis], sections: List[DocumentSection]) -> Dict:
        """Generate document summary statistics."""

        content_type_counts = {}
        topic_counts = {}
        total_compounds = set()

        for analysis in analyses:
            # Count content types
            content_type_counts[analysis.content_type] = content_type_counts.get(analysis.content_type, 0) + 1

            # Count topics
            topic_counts[analysis.chemistry_topic] = topic_counts.get(analysis.chemistry_topic, 0) + 1

            # Collect unique compounds
            total_compounds.update(analysis.key_compounds)

        return {
            'content_distribution': content_type_counts,
            'topic_distribution': topic_counts,
            'unique_compounds': list(total_compounds)[:20],  # Top 20
            'average_confidence': sum(a.confidence for a in analyses) / len(analyses) if analyses else 0,
            'total_sections': len(sections),
            'avg_chunks_per_section': len(analyses) / len(sections) if sections else 0
        }


if __name__ == "__main__":
    # Example usage
    analyzer = DocumentStructureAnalyzer()

    # Test with sample text
    sample_text = """
    Chapter 1: Introduction to Organic Chemistry

    Organic chemistry is the study of carbon compounds. Carbon forms four covalent bonds
    and can create complex molecular structures. The basic hydrocarbons include alkanes,
    alkenes, and alkynes.

    1.1 Alkanes

    Alkanes are saturated hydrocarbons with the general formula CnH2n+2. Methane (CH4)
    is the simplest alkane. The synthesis of alkanes can be achieved through several methods.

    Experimental Procedure:

    To synthesize methane, we react sodium acetate with soda lime at 300°C:
    CH3COONa + NaOH → CH4 + Na2CO3

    The reaction mechanism involves decarboxylation...
    """

    try:
        result = analyzer.analyze_document_structure(sample_text, "Organic Chemistry Sample")

        print(f"Analysis Results:")
        print(f"- Total chunks: {result['total_chunks']}")
        print(f"- Total sections: {result['total_sections']}")
        print(f"- Content distribution: {result['summary']['content_distribution']}")
        print(f"- Topic distribution: {result['summary']['topic_distribution']}")
        print(f"- Key compounds: {result['summary']['unique_compounds'][:10]}")

    except Exception as e:
        print(f"Analysis failed: {e}")