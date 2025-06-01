import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any

from ..lexicon import LexiconGenerator, Vocabulary
from ..sentence import TemplateManager, SentenceGenerator
from .corpus_evaluator import CorpusEvaluator


class SyntheticCorpusGenerator:
    """
    Main class for generating synthetic corpora with controlled statistical
    properties and semantic frame annotations for NLP tasks.
    """

    def __init__(self, vocab_sizes: Optional[Vocabulary] = None):
        """
        Initialize the corpus generator.

        Args:
            vocab_sizes: Optional Vocabulary instance with custom sizes
        """
        # Initialize components with semantic frame support
        self.lexicon = LexiconGenerator(vocab_sizes)
        self.templates = TemplateManager()
        self.sentence_generator = SentenceGenerator(self.lexicon, self.templates)
        self.evaluator = CorpusEvaluator()

        # Corpus storage
        self.corpus = []
        self.metadata = {}

    def generate_corpus(self,
                        num_sentences: int,
                        complexity_distribution: Optional[Dict[str, float]] = None,
                        semantic_frame_distribution: Optional[Dict[str, float]] = None,
                        include_annotations: bool = True) -> List[Dict[str, Any]]:
        """
        Generate a corpus with specified parameters and semantic frame diversity.

        Args:
            num_sentences: Number of sentences to generate
            complexity_distribution: Target complexity distribution
            semantic_frame_distribution: Target semantic frame distribution
            include_annotations: Whether to include full linguistic annotations

        Returns:
            List of generated sentence dictionaries with semantic annotations
        """
        # Default distributions
        if complexity_distribution is None:
            complexity_distribution = {"simple": 0.55, "medium": 0.35, "complex": 0.1}

        if semantic_frame_distribution is None:
            semantic_frame_distribution = {
                "transitive_action": 0.4,
                "intransitive_action": 0.25,
                "communication": 0.2,
                "motion": 0.15
            }

        print(f"Generating {num_sentences} sentences with semantic frame diversity...")

        # Generate initial corpus
        self.corpus = []
        frame_targets = self._calculate_frame_targets(num_sentences, semantic_frame_distribution)
        complexity_targets = self._calculate_complexity_targets(num_sentences, complexity_distribution)

        # Generate sentences by frame and complexity
        for frame_name, target_count in frame_targets.items():
            for complexity in ["simple", "medium", "complex"]:
                complexity_ratio = complexity_distribution[complexity]
                frame_complexity_count = int(target_count * complexity_ratio)

                for _ in range(frame_complexity_count):
                    sentence_data = self.sentence_generator.generate_sentence(
                        complexity=complexity,
                        include_metadata=include_annotations
                    )

                    # Add corpus-level metadata
                    sentence_data["corpus_metadata"] = {
                        "target_frame": frame_name,
                        "sentence_id": len(self.corpus),
                        "generation_timestamp": self._get_timestamp()
                    }

                    self.corpus.append(sentence_data)

                    if len(self.corpus) % 1000 == 0:
                        print(f"Generated {len(self.corpus)} sentences...")

        # Adjust corpus to exact target size and distributions
        self._adjust_corpus_distribution(
            complexity_distribution,
            semantic_frame_distribution,
            num_sentences
        )

        # Store generation metadata
        self.metadata = {
            "generation_params": {
                "num_sentences": num_sentences,
                "complexity_distribution": complexity_distribution,
                "semantic_frame_distribution": semantic_frame_distribution,
                "include_annotations": include_annotations
            },
            "actual_distributions": self._calculate_actual_distributions(),
            "lexicon_info": self.lexicon.export_lexicon_details(),
            "template_usage": self.templates.get_template_usage()
        }

        print(f"Corpus generation complete: {len(self.corpus)} sentences")
        return self.corpus

    def _calculate_frame_targets(self, num_sentences: int, frame_dist: Dict[str, float]) -> Dict[str, int]:
        """Calculate target counts for each semantic frame."""
        return {frame: int(num_sentences * ratio) for frame, ratio in frame_dist.items()}

    def _calculate_complexity_targets(self, num_sentences: int, complexity_dist: Dict[str, float]) -> Dict[str, int]:
        """Calculate target counts for each complexity level."""
        return {complexity: int(num_sentences * ratio) for complexity, ratio in complexity_dist.items()}

    def _adjust_corpus_distribution(self,
                                    complexity_dist: Dict[str, float],
                                    frame_dist: Dict[str, float],
                                    target_size: int):
        """Fine-tune corpus to match exact target distributions."""
        # Group by complexity and frame
        by_complexity = defaultdict(list)
        by_frame = defaultdict(list)

        for i, item in enumerate(self.corpus):
            if "metadata" in item:
                complexity = item["metadata"].get("complexity", "simple")
                frame = item["metadata"].get("frame", "transitive_action")
                by_complexity[complexity].append(i)
                by_frame[frame].append(i)

        # Adjust for target size
        if len(self.corpus) > target_size:
            excess = len(self.corpus) - target_size
            indices_to_remove = random.sample(range(len(self.corpus)), excess)
            self.corpus = [item for i, item in enumerate(self.corpus) if i not in indices_to_remove]
        elif len(self.corpus) < target_size:
            deficit = target_size - len(self.corpus)
            for _ in range(deficit):
                # Generate additional sentences with balanced distribution
                complexity = random.choices(
                    list(complexity_dist.keys()),
                    weights=list(complexity_dist.values())
                )[0]

                sentence_data = self.sentence_generator.generate_sentence(complexity=complexity)
                sentence_data["corpus_metadata"] = {
                    "sentence_id": len(self.corpus),
                    "generation_timestamp": self._get_timestamp(),
                    "adjustment_generation": True
                }
                self.corpus.append(sentence_data)

    def _calculate_actual_distributions(self) -> Dict[str, Dict[str, float]]:
        """Calculate actual distributions in the generated corpus."""
        total = len(self.corpus)

        complexity_counts = Counter()
        frame_counts = Counter()

        for item in self.corpus:
            if "metadata" in item:
                complexity_counts[item["metadata"].get("complexity", "unknown")] += 1
                frame_counts[item["metadata"].get("frame", "unknown")] += 1

        return {
            "complexity": {k: v / total for k, v in complexity_counts.items()},
            "semantic_frames": {k: v / total for k, v in frame_counts.items()}
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        import datetime
        return datetime.datetime.now().isoformat()

    def save_corpus(self,
                    output_file: str,
                    format_type: str = "full",
                    include_stats: bool = False):
        """
        Save the generated corpus in various formats.

        Args:
            output_file: Output filename
            format_type: Output format ("full", "sentences_only", "semantic_only", "conll")
            include_stats: Whether to include corpus statistics
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        # Prepare output based on format
        if format_type == "full":
            # Convert corpus to JSON-serializable format
            serializable_corpus = self._make_json_serializable(self.corpus)
            output_data = {
                "corpus": serializable_corpus,
                "metadata": self.metadata
            }
            if include_stats:
                output_data["statistics"] = self.evaluator.analyze_corpus(self.corpus)

        elif format_type == "sentences_only":
            output_data = [item["sentence"] for item in self.corpus]

        elif format_type == "semantic_only":
            output_data = []
            for item in self.corpus:
                semantic_item = {
                    "sentence": item["sentence"],
                    "semantic_roles": item.get("semantic_roles", {}),
                    "semantics": item.get("semantics", ""),
                    "frame": item.get("metadata", {}).get("frame", "")
                }
                output_data.append(semantic_item)

        elif format_type == "conll":
            self._save_conll_format(output_file)
            return

        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        # Save JSON output with custom encoder
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, cls=CorpusJSONEncoder)

        print(f"Corpus saved to '{output_file}' in {format_type} format")

    def _save_conll_format(self, output_file: str):
        """Save corpus in CoNLL-style format for NLP tools."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(self.corpus):
                # Write sentence comment
                f.write(f"# sent_id = {i + 1}\n")
                f.write(f"# text = {item['sentence']}\n")

                if "semantic_roles" in item:
                    f.write(f"# semantic_frame = {item.get('metadata', {}).get('frame', '')}\n")

                # Write token annotations
                words = item["sentence"].split()
                annotations = item.get("linguistic_annotations", {})
                pos_tags = annotations.get("pos_tags", ["_"] * len(words))

                for j, (word, pos) in enumerate(zip(words, pos_tags)):
                    # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
                    semantic_role = self._get_word_semantic_role(word, item.get("semantic_roles", {}))
                    misc = f"SemRole={semantic_role}" if semantic_role else "_"

                    f.write(f"{j + 1}\t{word}\t{word}\t{pos}\t_\t_\t_\t_\t_\t{misc}\n")

                f.write("\n")  # Empty line between sentences

    def _get_word_semantic_role(self, word: str, semantic_roles: Dict) -> Optional[str]:
        """Get semantic role for a specific word."""
        for arg, role_info in semantic_roles.items():
            if role_info.get("word") == word:
                return role_info.get("role")
        return None

    def evaluate_corpus(self) -> Dict[str, Any]:
        """
        Comprehensive corpus evaluation with semantic frame analysis.

        Returns:
            Dictionary with evaluation metrics and recommendations
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")

        # Standard corpus evaluation
        stats = self.evaluator.analyze_corpus(self.corpus)
        suitability = self.evaluator.evaluate_next_token_prediction_suitability()

        # Semantic frame analysis
        frame_analysis = self._analyze_semantic_frames()

        print(f"Corpus evaluation complete:")
        print(f"  Suitability score: {suitability['suitability_score']:.2f} ({suitability['suitability_category']})")
        print(f"  Semantic frame diversity: {frame_analysis['frame_diversity']:.3f}")
        print(f"  Role coverage: {frame_analysis['role_coverage']:.1%}")

        if suitability["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(suitability["recommendations"], 1):
                print(f"  {i}. {rec}")

        return {
            "statistics": stats,
            "suitability": suitability,
            "semantic_analysis": frame_analysis,
            "metadata": self.metadata
        }

    def _analyze_semantic_frames(self) -> Dict[str, Any]:
        """Analyze semantic frame distribution and coverage."""
        frame_counts = Counter()
        role_counts = Counter()
        total_sentences = len(self.corpus)

        for item in self.corpus:
            # Count frames
            frame = item.get("metadata", {}).get("frame", "unknown")
            frame_counts[frame] += 1

            # Count roles
            for role_info in item.get("semantic_roles", {}).values():
                role_counts[role_info.get("role", "unknown")] += 1

        # Calculate diversity metrics
        frame_diversity = len(frame_counts) / max(1, total_sentences)  # Unique frames per sentence
        role_coverage = len(role_counts) / 9  # Coverage of standard semantic roles (9 total)

        return {
            "frame_distribution": {k: v / total_sentences for k, v in frame_counts.items()},
            "role_distribution": dict(role_counts),
            "frame_diversity": frame_diversity,
            "role_coverage": role_coverage,
            "unique_frames": len(frame_counts),
            "unique_roles": len(role_counts)
        }

    def generate_subsets(self, subset_configs: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        Generate specialized subsets of the corpus for different tasks.

        Args:
            subset_configs: Configuration for each subset
                Example: {
                    "simple_transitive": {"complexity": ["simple"], "frames": ["transitive_action"]},
                    "complex_communication": {"complexity": ["complex"], "frames": ["communication"]}
                }

        Returns:
            Dictionary mapping subset names to sentence lists
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")

        subsets = {}

        for subset_name, config in subset_configs.items():
            target_complexities = config.get("complexity", [])
            target_frames = config.get("frames", [])
            max_size = config.get("max_size", None)

            filtered_sentences = []

            for item in self.corpus:
                metadata = item.get("metadata", {})
                complexity = metadata.get("complexity", "")
                frame = metadata.get("frame", "")

                # Check if sentence matches criteria
                complexity_match = not target_complexities or complexity in target_complexities
                frame_match = not target_frames or frame in target_frames

                if complexity_match and frame_match:
                    filtered_sentences.append(item)

                    if max_size and len(filtered_sentences) >= max_size:
                        break

            subsets[subset_name] = filtered_sentences
            print(f"Generated subset '{subset_name}': {len(filtered_sentences)} sentences")

    def _make_json_serializable(self, corpus: List[Dict]) -> List[Dict]:
        """Convert corpus to JSON-serializable format."""
        serializable_corpus = []

        for item in corpus:
            serializable_item = {}

            for key, value in item.items():
                if key == "linguistic_annotations":
                    # Convert LinguisticAnnotation to dict
                    if hasattr(value, '__dict__'):
                        serializable_item[key] = value.__dict__
                    elif hasattr(value, '_asdict'):  # namedtuple
                        serializable_item[key] = value._asdict()
                    else:
                        serializable_item[key] = self._convert_to_serializable(value)
                else:
                    serializable_item[key] = self._convert_to_serializable(value)

            serializable_corpus.append(serializable_item)

        return serializable_corpus

    def _convert_to_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        # Handle LinguisticAnnotation specifically
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        # Handle enums first (before checking __dict__)
        elif hasattr(obj, 'value'):
            return obj.value
        # Handle mappingproxy objects (from enum internals)
        elif str(type(obj)) == "<class 'mappingproxy'>":
            return dict(obj)
        # Handle other dataclasses
        elif hasattr(obj, '__dict__') and hasattr(obj, '__dataclass_fields__'):
            return {k: self._convert_to_serializable(v) for k, v in obj.__dict__.items()}
        # Handle general objects with __dict__
        elif hasattr(obj, '__dict__'):
            return {k: self._convert_to_serializable(v) for k, v in obj.__dict__.items()}
        # Handle namedtuples
        elif hasattr(obj, '_asdict'):
            return self._convert_to_serializable(obj._asdict())
        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        # Handle lists and tuples recursively
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        # Handle sets
        elif isinstance(obj, set):
            return [self._convert_to_serializable(item) for item in obj]
        # Handle numpy arrays
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        # Return as-is for basic types
        else:
            return obj


class CorpusJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for corpus data structures."""

    def default(self, obj):
        # Handle objects with to_dict method (like LinguisticAnnotation)
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        # Handle enums first (before checking __dict__)
        elif hasattr(obj, 'value'):
            return obj.value
        # Handle mappingproxy objects (from enum internals)
        elif str(type(obj)) == "<class 'mappingproxy'>":
            return dict(obj)
        # Handle dataclasses
        elif hasattr(obj, '__dict__') and hasattr(obj, '__dataclass_fields__'):
            return obj.__dict__
        # Handle general objects with __dict__
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        # Handle namedtuples
        elif hasattr(obj, '_asdict'):
            return obj._asdict()
        # Handle numpy types
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
        # Default behavior
        return super().default(obj)

# import json
# import os
# import random
# from collections import defaultdict, Counter
# from typing import Dict, List, Optional, Any
#
# from ..lexicon import LexiconGenerator, Vocabulary
# from ..sentence import TemplateManager, SentenceGenerator
# from .corpus_evaluator import CorpusEvaluator
#
#
# class SyntheticCorpusGenerator:
#     """
#     Main class for generating synthetic corpora with controlled statistical
#     properties and semantic frame annotations for NLP tasks.
#     """
#
#     def __init__(self, vocab_sizes: Optional[Vocabulary] = None):
#         """
#         Initialize the corpus generator.
#
#         Args:
#             vocab_sizes: Optional Vocabulary instance with custom sizes
#         """
#         # Initialize components with semantic frame support
#         self.lexicon = LexiconGenerator(vocab_sizes)
#         self.templates = TemplateManager()
#         self.sentence_generator = SentenceGenerator(self.lexicon, self.templates)
#         self.evaluator = CorpusEvaluator()
#
#         # Corpus storage
#         self.corpus = []
#         self.metadata = {}
#
#     def generate_corpus(self,
#                         num_sentences: int,
#                         complexity_distribution: Optional[Dict[str, float]] = None,
#                         semantic_frame_distribution: Optional[Dict[str, float]] = None,
#                         include_annotations: bool = True) -> List[Dict[str, Any]]:
#         """
#         Generate a corpus with specified parameters and semantic frame diversity.
#
#         Args:
#             num_sentences: Number of sentences to generate
#             complexity_distribution: Target complexity distribution
#             semantic_frame_distribution: Target semantic frame distribution
#             include_annotations: Whether to include full linguistic annotations
#
#         Returns:
#             List of generated sentence dictionaries with semantic annotations
#         """
#         # Default distributions
#         if complexity_distribution is None:
#             complexity_distribution = {"simple": 0.55, "medium": 0.35, "complex": 0.1}
#
#         if semantic_frame_distribution is None:
#             semantic_frame_distribution = {
#                 "transitive_action": 0.4,
#                 "intransitive_action": 0.25,
#                 "communication": 0.2,
#                 "motion": 0.15
#             }
#
#         print(f"Generating {num_sentences} sentences with semantic frame diversity...")
#
#         # Generate initial corpus
#         self.corpus = []
#         frame_targets = self._calculate_frame_targets(num_sentences, semantic_frame_distribution)
#         complexity_targets = self._calculate_complexity_targets(num_sentences, complexity_distribution)
#
#         # Generate sentences by frame and complexity
#         for frame_name, target_count in frame_targets.items():
#             for complexity in ["simple", "medium", "complex"]:
#                 complexity_ratio = complexity_distribution[complexity]
#                 frame_complexity_count = int(target_count * complexity_ratio)
#
#                 for _ in range(frame_complexity_count):
#                     sentence_data = self.sentence_generator.generate_sentence(
#                         complexity=complexity,
#                         include_metadata=include_annotations
#                     )
#
#                     # Add corpus-level metadata
#                     sentence_data["corpus_metadata"] = {
#                         "target_frame": frame_name,
#                         "sentence_id": len(self.corpus),
#                         "generation_timestamp": self._get_timestamp()
#                     }
#
#                     self.corpus.append(sentence_data)
#
#                     if len(self.corpus) % 1000 == 0:
#                         print(f"Generated {len(self.corpus)} sentences...")
#
#         # Adjust corpus to exact target size and distributions
#         self._adjust_corpus_distribution(
#             complexity_distribution,
#             semantic_frame_distribution,
#             num_sentences
#         )
#
#         # Store generation metadata
#         self.metadata = {
#             "generation_params": {
#                 "num_sentences": num_sentences,
#                 "complexity_distribution": complexity_distribution,
#                 "semantic_frame_distribution": semantic_frame_distribution,
#                 "include_annotations": include_annotations
#             },
#             "actual_distributions": self._calculate_actual_distributions(),
#             "lexicon_info": self.lexicon.export_lexicon_details(),
#             "template_usage": self.templates.get_template_usage()
#         }
#
#         print(f"Corpus generation complete: {len(self.corpus)} sentences")
#         return self.corpus
#
#     def _calculate_frame_targets(self, num_sentences: int, frame_dist: Dict[str, float]) -> Dict[str, int]:
#         """Calculate target counts for each semantic frame."""
#         return {frame: int(num_sentences * ratio) for frame, ratio in frame_dist.items()}
#
#     def _calculate_complexity_targets(self, num_sentences: int, complexity_dist: Dict[str, float]) -> Dict[str, int]:
#         """Calculate target counts for each complexity level."""
#         return {complexity: int(num_sentences * ratio) for complexity, ratio in complexity_dist.items()}
#
#     def _adjust_corpus_distribution(self,
#                                     complexity_dist: Dict[str, float],
#                                     frame_dist: Dict[str, float],
#                                     target_size: int):
#         """Fine-tune corpus to match exact target distributions."""
#         # Group by complexity and frame
#         by_complexity = defaultdict(list)
#         by_frame = defaultdict(list)
#
#         for i, item in enumerate(self.corpus):
#             if "metadata" in item:
#                 complexity = item["metadata"].get("complexity", "simple")
#                 frame = item["metadata"].get("frame", "transitive_action")
#                 by_complexity[complexity].append(i)
#                 by_frame[frame].append(i)
#
#         # Adjust for target size
#         if len(self.corpus) > target_size:
#             excess = len(self.corpus) - target_size
#             indices_to_remove = random.sample(range(len(self.corpus)), excess)
#             self.corpus = [item for i, item in enumerate(self.corpus) if i not in indices_to_remove]
#         elif len(self.corpus) < target_size:
#             deficit = target_size - len(self.corpus)
#             for _ in range(deficit):
#                 # Generate additional sentences with balanced distribution
#                 complexity = random.choices(
#                     list(complexity_dist.keys()),
#                     weights=list(complexity_dist.values())
#                 )[0]
#
#                 sentence_data = self.sentence_generator.generate_sentence(complexity=complexity)
#                 sentence_data["corpus_metadata"] = {
#                     "sentence_id": len(self.corpus),
#                     "generation_timestamp": self._get_timestamp(),
#                     "adjustment_generation": True
#                 }
#                 self.corpus.append(sentence_data)
#
#     def _calculate_actual_distributions(self) -> Dict[str, Dict[str, float]]:
#         """Calculate actual distributions in the generated corpus."""
#         total = len(self.corpus)
#
#         complexity_counts = Counter()
#         frame_counts = Counter()
#
#         for item in self.corpus:
#             if "metadata" in item:
#                 complexity_counts[item["metadata"].get("complexity", "unknown")] += 1
#                 frame_counts[item["metadata"].get("frame", "unknown")] += 1
#
#         return {
#             "complexity": {k: v / total for k, v in complexity_counts.items()},
#             "semantic_frames": {k: v / total for k, v in frame_counts.items()}
#         }
#
#     def _get_timestamp(self) -> str:
#         """Get current timestamp for metadata."""
#         import datetime
#         return datetime.datetime.now().isoformat()
#
#     def save_corpus(self,
#                     output_file: str,
#                     format_type: str = "full",
#                     include_stats: bool = False):
#         """
#         Save the generated corpus in various formats.
#
#         Args:
#             output_file: Output filename
#             format_type: Output format ("full", "sentences_only", "semantic_only", "conll")
#             include_stats: Whether to include corpus statistics
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         # Create output directory if needed
#         os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
#
#         # Prepare output based on format
#         if format_type == "full":
#             output_data = {
#                 "corpus": self.corpus,
#                 "metadata": self.metadata
#             }
#             if include_stats:
#                 output_data["statistics"] = self.evaluator.analyze_corpus(self.corpus)
#
#         elif format_type == "sentences_only":
#             output_data = [item["sentence"] for item in self.corpus]
#
#         elif format_type == "semantic_only":
#             output_data = []
#             for item in self.corpus:
#                 semantic_item = {
#                     "sentence": item["sentence"],
#                     "semantic_roles": item.get("semantic_roles", {}),
#                     "semantics": item.get("semantics", ""),
#                     "frame": item.get("metadata", {}).get("frame", "")
#                 }
#                 output_data.append(semantic_item)
#
#         elif format_type == "conll":
#             self._save_conll_format(output_file)
#             return
#
#         else:
#             raise ValueError(f"Unknown format_type: {format_type}")
#
#         # Save JSON output
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, indent=2, ensure_ascii=False)
#
#         print(f"Corpus saved to '{output_file}' in {format_type} format")
#
#     def _save_conll_format(self, output_file: str):
#         """Save corpus in CoNLL-style format for NLP tools."""
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for i, item in enumerate(self.corpus):
#                 # Write sentence comment
#                 f.write(f"# sent_id = {i + 1}\n")
#                 f.write(f"# text = {item['sentence']}\n")
#
#                 if "semantic_roles" in item:
#                     f.write(f"# semantic_frame = {item.get('metadata', {}).get('frame', '')}\n")
#
#                 # Write token annotations
#                 words = item["sentence"].split()
#                 annotations = item.get("linguistic_annotations", {})
#                 pos_tags = annotations.get("pos_tags", ["_"] * len(words))
#
#                 for j, (word, pos) in enumerate(zip(words, pos_tags)):
#                     # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
#                     semantic_role = self._get_word_semantic_role(word, item.get("semantic_roles", {}))
#                     misc = f"SemRole={semantic_role}" if semantic_role else "_"
#
#                     f.write(f"{j + 1}\t{word}\t{word}\t{pos}\t_\t_\t_\t_\t_\t{misc}\n")
#
#                 f.write("\n")  # Empty line between sentences
#
#     def _get_word_semantic_role(self, word: str, semantic_roles: Dict) -> Optional[str]:
#         """Get semantic role for a specific word."""
#         for arg, role_info in semantic_roles.items():
#             if role_info.get("word") == word:
#                 return role_info.get("role")
#         return None
#
#     def evaluate_corpus(self) -> Dict[str, Any]:
#         """
#         Comprehensive corpus evaluation with semantic frame analysis.
#
#         Returns:
#             Dictionary with evaluation metrics and recommendations
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         # Standard corpus evaluation
#         stats = self.evaluator.analyze_corpus(self.corpus)
#         suitability = self.evaluator.evaluate_next_token_prediction_suitability()
#
#         # Semantic frame analysis
#         frame_analysis = self._analyze_semantic_frames()
#
#         print(f"Corpus evaluation complete:")
#         print(f"  Suitability score: {suitability['suitability_score']:.2f} ({suitability['suitability_category']})")
#         print(f"  Semantic frame diversity: {frame_analysis['frame_diversity']:.3f}")
#         print(f"  Role coverage: {frame_analysis['role_coverage']:.1%}")
#
#         if suitability["recommendations"]:
#             print("\nRecommendations:")
#             for i, rec in enumerate(suitability["recommendations"], 1):
#                 print(f"  {i}. {rec}")
#
#         return {
#             "statistics": stats,
#             "suitability": suitability,
#             "semantic_analysis": frame_analysis,
#             "metadata": self.metadata
#         }
#
#     def _analyze_semantic_frames(self) -> Dict[str, Any]:
#         """Analyze semantic frame distribution and coverage."""
#         frame_counts = Counter()
#         role_counts = Counter()
#         total_sentences = len(self.corpus)
#
#         for item in self.corpus:
#             # Count frames
#             frame = item.get("metadata", {}).get("frame", "unknown")
#             frame_counts[frame] += 1
#
#             # Count roles
#             for role_info in item.get("semantic_roles", {}).values():
#                 role_counts[role_info.get("role", "unknown")] += 1
#
#         # Calculate diversity metrics
#         frame_diversity = len(frame_counts) / max(1, total_sentences)  # Unique frames per sentence
#         role_coverage = len(role_counts) / 9  # Coverage of standard semantic roles (9 total)
#
#         return {
#             "frame_distribution": {k: v / total_sentences for k, v in frame_counts.items()},
#             "role_distribution": dict(role_counts),
#             "frame_diversity": frame_diversity,
#             "role_coverage": role_coverage,
#             "unique_frames": len(frame_counts),
#             "unique_roles": len(role_counts)
#         }
#
#     def generate_subsets(self, subset_configs: Dict[str, Dict]) -> Dict[str, List[Dict]]:
#         """
#         Generate specialized subsets of the corpus for different tasks.
#
#         Args:
#             subset_configs: Configuration for each subset
#                 Example: {
#                     "simple_transitive": {"complexity": ["simple"], "frames": ["transitive_action"]},
#                     "complex_communication": {"complexity": ["complex"], "frames": ["communication"]}
#                 }
#
#         Returns:
#             Dictionary mapping subset names to sentence lists
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         subsets = {}
#
#         for subset_name, config in subset_configs.items():
#             target_complexities = config.get("complexity", [])
#             target_frames = config.get("frames", [])
#             max_size = config.get("max_size", None)
#
#             filtered_sentences = []
#
#             for item in self.corpus:
#                 metadata = item.get("metadata", {})
#                 complexity = metadata.get("complexity", "")
#                 frame = metadata.get("frame", "")
#
#                 # Check if sentence matches criteria
#                 complexity_match = not target_complexities or complexity in target_complexities
#                 frame_match = not target_frames or frame in target_frames
#
#                 if complexity_match and frame_match:
#                     filtered_sentences.append(item)
#
#                     if max_size and len(filtered_sentences) >= max_size:
#                         break
#
#             subsets[subset_name] = filtered_sentences
#             print(f"Generated subset '{subset_name}': {len(filtered_sentences)} sentences")
#
#         return subsets

# from absynth.lexicon.lexicon_generator import LexiconGenerator
# from absynth.sentence.frame_manager import FrameManager
# from absynth.sentence.sentence_generator import SentenceGenerator
# from absynth.corpus.corpus_evaluator import CorpusEvaluator
# import json
# import os
#
#
# class SyntheticCorpusGenerator:
#     """
#     Main class for generating synthetic corpora with controlled statistical
#     properties for next token prediction and other NLP tasks.
#     """
#
#     def __init__(self, vocab_sizes=None):
#         """
#         Initialize the corpus generator with default or custom vocabulary sizes.
#
#         Args:
#             vocab_sizes: Optional dictionary mapping word categories to vocab sizes
#         """
#         # Default vocabulary sizes if not provided
#         self.vocab_sizes = vocab_sizes or {
#             "noun": 200,
#             "transitive_verb": 30,
#             "intransitive_verb": 30,
#             "communication_verb": 15,
#             "motion_verb": 15,
#             "change_verb": 15,
#             "adjective": 25,
#             "adverb": 15,
#             "location": 100,
#             "temporal": 20,
#         }
#
#         # Initialize components
#         self.lexicon = LexiconGenerator(self.vocab_sizes)
#         self.templates = FrameManager()
#         self.sentence_generator = SentenceGenerator(self.lexicon, self.templates)
#         self.evaluator = CorpusEvaluator()
#         self.corpus = []
#
#     def generate_corpus(self, num_sentences, complexity_distribution=None):
#         """
#         Generate a corpus with specified number of sentences and complexity distribution.
#
#         Args:
#             num_sentences: Number of sentences to generate
#             complexity_distribution: Optional dict with target distribution
#                 e.g., {"simple": 0.55, "medium": 0.35, "complex": 0.1}
#
#         Returns:
#             List of generated sentence dictionaries
#         """
#         # Default complexity distribution if not provided
#         complexity_distribution = complexity_distribution or {
#             "simple": 0.55, "medium": 0.35, "complex": 0.1
#         }
#
#         self.corpus = []
#
#         # Generate sentences
#         print(f"Generating {num_sentences} sentences...")
#         for i in range(num_sentences):
#             if i % 1000 == 0 and i > 0:
#                 print(f"Generated {i} sentences...")
#
#             sentence = self.sentence_generator.generate_sentence()
#             self.corpus.append(sentence)
#
#         # Adjust corpus to match target distribution
#         self._adjust_corpus_distribution(complexity_distribution, num_sentences)
#
#         # Update evaluator with the generated corpus
#         self.evaluator.corpus = self.corpus
#
#         return self.corpus
#
#     def _adjust_corpus_distribution(self, target_dist, num_sentences):
#         """
#         Adjust corpus to match target complexity distribution.
#
#         Args:
#             target_dist: Dictionary with target distribution percentages
#             num_sentences: Target number of sentences
#         """
#         import random
#         from collections import defaultdict
#
#         # Group sentences by complexity
#         by_complexity = defaultdict(list)
#         for i, item in enumerate(self.corpus):
#             complexity = item["metadata"]["complexity"]
#             by_complexity[complexity].append(i)
#
#         # Calculate current counts and target counts
#         current_counts = {k: len(v) for k, v in by_complexity.items()}
#         target_counts = {k: int(v * num_sentences) for k, v in target_dist.items()}
#
#         # Add sentences for complexities that need more
#         for complexity, target in target_counts.items():
#             current = current_counts.get(complexity, 0)
#
#             if current < target:
#                 # Generate additional sentences of this complexity
#                 for _ in range(target - current):
#                     sentence = self.sentence_generator.generate_sentence(complexity)
#                     self.corpus.append(sentence)
#
#         # Trim corpus if too large
#         if len(self.corpus) > num_sentences:
#             # Recalculate groups
#             by_complexity = defaultdict(list)
#             for i, item in enumerate(self.corpus):
#                 complexity = item["metadata"]["complexity"]
#                 by_complexity[complexity].append(i)
#
#             current_counts = {k: len(v) for k, v in by_complexity.items()}
#
#             # Calculate how many to remove from each group
#             to_remove = {}
#             for complexity in by_complexity:
#                 current = current_counts.get(complexity, 0)
#                 target = target_counts.get(complexity, 0)
#                 to_remove[complexity] = max(0, current - target)
#
#             # Create list of indices to remove
#             indices_to_remove = []
#             for complexity, count in to_remove.items():
#                 indices = by_complexity[complexity]
#                 # Randomly select indices to remove
#                 if count > 0 and indices:
#                     selected = random.sample(indices, min(count, len(indices)))
#                     indices_to_remove.extend(selected)
#
#             # Create new corpus without removed indices
#             self.corpus = [item for i, item in enumerate(self.corpus) if i not in indices_to_remove]
#
#     def save_corpus(self, output_file, include_stats=False):
#         """
#         Save the generated corpus to a JSON file.
#
#         Args:
#             output_file: Output filename
#             include_stats: Whether to include corpus statistics
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         output = {"corpus": self.corpus}
#
#         if include_stats:
#             output["statistics"] = self.evaluator.analyze_corpus()
#         else:
#             # Remove statistics from output
#             for item in self.corpus:
#                 item.pop("metadata", None)
#             output = output["corpus"]
#
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
#
#         with open(output_file, 'w') as f:
#             json.dump(output, f, indent=2)
#
#         print(f"Corpus with {len(self.corpus)} sentences saved to '{output_file}'")
#
#     def evaluate_corpus(self):
#         """
#         Evaluate the generated corpus for next token prediction suitability.
#
#         Returns:
#             Dictionary with evaluation metrics
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         # Analyze corpus statistics
#         self.evaluator.corpus = self.corpus
#         stats = self.evaluator.analyze_corpus()
#
#         # Evaluate suitability for next token prediction
#         suitability = self.evaluator.evaluate_next_token_prediction_suitability()
#
#         print(f"Corpus evaluation complete. Suitability score: {suitability['suitability_score']:.2f} ({suitability['suitability_category']})")
#
#         if suitability["recommendations"]:
#             print("\nRecommendations for improvement:")
#             for i, rec in enumerate(suitability["recommendations"], 1):
#                 print(f"{i}. {rec}")
#
#         return {
#             "statistics": stats,
#             "suitability": suitability
#         }
#
#     def plot_statistics(self, output_dir=None):
#         """
#         Generate plots of corpus statistics.
#
#         Args:
#             output_dir: Optional directory to save plots
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         self.evaluator.corpus = self.corpus
#         self.evaluator.plot_statistics(output_dir)
#
#     def generate_evaluation_report(self, output_file):
#         """
#         Generate and save a comprehensive evaluation report.
#
#         Args:
#             output_file: Output filename for the report
#
#         Returns:
#             Dictionary with evaluation report
#         """
#         if not self.corpus:
#             raise ValueError("No corpus generated yet. Call generate_corpus() first.")
#
#         self.evaluator.corpus = self.corpus
#         report = self.evaluator.generate_report(output_file)
#
#         return report


# def main():
#     """Command-line interface for synthetic corpus generation."""
#     parser = argparse.ArgumentParser(description='Generate synthetic corpus for NLP tasks')
#     parser.add_argument('--size', type=int, default=10000, help='Number of sentences')
#     parser.add_argument('--output', type=str, default='synthetic_corpus.json', help='Output file')
#     parser.add_argument('--simple', type=float, default=0.55, help='Proportion of simple sentences')
#     parser.add_argument('--medium', type=float, default=0.35, help='Proportion of medium sentences')
#     parser.add_argument('--complex', type=float, default=0.1, help='Proportion of complex sentences')
#     parser.add_argument('--plots', action='store_true', help='Generate evaluation plots')
#     parser.add_argument('--plots-dir', type=str, default='plots', help='Directory for plots')
#     parser.add_argument('--report', type=str, help='Output file for evaluation report')
#
#     args = parser.parse_args()
#
#     # Normalize complexity proportions
#     total = args.simple + args.medium + args.complex
#     complexity_dist = {
#         "simple": args.simple / total,
#         "medium": args.medium / total,
#         "complex": args.complex / total
#     }
#
#     # Create and run corpus generator
#     generator = SyntheticCorpusGenerator()
#     generator.generate_corpus(args.size, complexity_dist)
#     generator.save_corpus(args.output)
#
#     # Evaluate corpus
#     evaluation = generator.evaluate_corpus()
#
#     # Generate plots if requested
#     if args.plots:
#         os.makedirs(args.plots_dir, exist_ok=True)
#         generator.plot_statistics(args.plots_dir)
#
#     # Generate report if requested
#     if args.report:
#         generator.generate_evaluation_report(args.report)
#
#
# if __name__ == "__main__":
#     main()