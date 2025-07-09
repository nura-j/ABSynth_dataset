import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Sequence

from ..lexicon import LexiconGenerator, Vocabulary
from ..sentence import FrameManager, SentenceGenerator
from .. import SemanticFrame
from .corpus_evaluator import CorpusEvaluator


class SynthCorpus(Sequence[Dict[str, Any]]):
    def __init__(self, corpus_data: List[Dict[str, Any]], metadata: dict, statistics: dict[str, Any] = None):
        self._corpus = corpus_data
        self._metadata = metadata
        self._statistics = statistics

    def __iter__(self):
        iter(self._corpus)

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, item) -> Dict[str, Any]:
        return self._corpus[item]

    @property
    def metadata(self):
        return self._metadata

    @property
    def statistics(self):
        return self._statistics

    @property
    def sentences(self) -> List[str]:
        return [item["sentence"] for item in self._corpus]

    @property
    def semantic_annotations(self) -> List[Dict[str, Any]]:
        output_data = list()
        for item in self._corpus:
            semantic_item = {
                "sentence": item["sentence"],
                "semantic_roles": item.get("semantic_roles", {}),
                "semantics": item.get("semantics", ""),
                "frame": item.get("metadata", {}).get("frame", "")
            }
            output_data.append(semantic_item)

        return output_data

    @property
    def complexity_distribution(self) -> Dict[str, float]:
        return self._metadata["generation_params"]["complexity_distribution"]

    @property
    def semantic_frame_distribution(self) -> Dict[str, float]:
        return self._metadata["generation_params"]["semantic_frame_distribution"]

    @property
    def include_annotations(self) -> bool:
        return self._metadata["generation_params"]["include_annotations"]

    def corpus_data(self) -> List[Dict[str, Any]]:
        return self._corpus

    def save(self, path: str, include_stats: bool = False, indent: None | int = None):
        # Create output directory if needed
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        stats = CorpusEvaluator().analyze_corpus(self._corpus) if include_stats else dict()

        with open(path, "w") as out_file:
            json.dump({"corpus": self._corpus, "metadata": self._metadata, "statistics": stats}, out_file,
                      indent=indent, default=lambda x: x.value)

    def export(self, path: str, format: str = "sentences_only", indent: None | int = None):
        # Create output directory if needed
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        exp_method = {
            "sentences_only": self.__class__.sentences,
            "semantic_annotations": self.__class__.semantic_annotations
        }

        if (format in exp_method):
            with open(path, "w") as out_file:
                json.dump(exp_method[format].__get__(self), out_file, indent=indent, default=lambda x: x.value)

    @staticmethod
    def load(path: str):
        with open(path) as in_file:
            data = json.load(in_file)
        corpus = SynthCorpus(data["corpus"], data["metadata"], data["statistics"])

        return corpus


class SyntheticCorpusGenerator:
    """
    Main class for generating synthetic corpora with controlled statistical
    properties and semantic frame annotations for NLP tasks.
    """

    def __init__(self, vocab_sizes: Optional[Vocabulary] = None,
                 num_clusters: int = 5,
                 additional_semantic_frames: Optional[List[SemanticFrame]] = None,
                 zipfian_alpha: float = 1.05,
                 error_bias: float = 0.0000001,
                 random_seed: Optional[int] = None,
                 frames: Optional[List[SemanticFrame]] = None,
                 lexicon: Optional[LexiconGenerator] = None,
                 sentence_generator: Optional[SentenceGenerator] = None,
                 ):
        """
        Initialize the corpus generator.

        Args:
            vocab_sizes: Optional Vocabulary instance with custom sizes
            num_clusters: Number of clusters for Zipfian distribution
            additional_semantic_frames: Optional list of additional semantic frames
            zipfian_alpha: Zipfian distribution parameter for word frequency
            error_bias: Small bias to avoid zero probabilities in Zipfian distribution
            random_seed: Optional random seed for reproducibility
            frames: Optional list of semantic frames to use in the corpus
        """
        # Initialize components with semantic frame support
        self.lexicon = lexicon if lexicon else LexiconGenerator(vocab_sizes,
                                                num_clusters=num_clusters,
                                                additional_semantic_frames=additional_semantic_frames,
                                                zipfian_alpha=zipfian_alpha,
                                                error_bias=error_bias,
                                                random_seed=random_seed)

        self.frames = FrameManager(self.lexicon.semantic_frames if frames is None else frames)
        self.sentence_generator = sentence_generator if sentence_generator else SentenceGenerator(self.lexicon, self.frames)
        self.evaluator = CorpusEvaluator()

    def __call__(self, *args, **kwargs):
        return self.generate_corpus(*args, **kwargs)

    def generate_corpus(self,
                        num_sentences: int,
                        complexity_distribution: Optional[Dict[str, float]] = None,
                        semantic_frame_distribution: Optional[Dict[str, float]] = None,
                        frames_weights: Optional[Dict[str, float]] = None,
                        include_annotations: bool = True) -> SynthCorpus:
        """
        Generate a corpus with specified parameters and semantic frame diversity.

        Args:
            num_sentences: Number of sentences to generate
            complexity_distribution: Target complexity distribution
            semantic_frame_distribution: Target semantic frame distribution
            frames_weights: Custom weights for specific templates (e.g., {"S_V_O": 0.5, "S_V": 0.3})
            include_annotations: Whether to include full linguistic annotations

        Returns:
            List of generated sentence dictionaries with semantic annotations
        """
        # Default distributions
        if complexity_distribution is None:
            complexity_distribution = {"simple": 0.55, "medium": 0.35, "complex": 0.1}

        if semantic_frame_distribution is None:
            # get the deftault semantic frame keys from the FrameManager
            semantic_frame_distribution = self.frames.get_default_semantic_frames_distribution()

        # Set custom template weights if provided
        if frames_weights:
            self.frames.set_template_weights(frames_weights)
            print(f"Applied custom template weights: {frames_weights}")

        print(f"Generating {num_sentences} sentences with semantic frame diversity...")

        # Generate initial corpus
        corpus = []
        frame_targets = self.calculate_frame_targets(num_sentences, semantic_frame_distribution)
        complexity_targets = self.calculate_complexity_targets(num_sentences, complexity_distribution)

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
                        "sentence_id": len(corpus),
                    }

                    corpus.append(sentence_data)

                    if len(corpus) % 1000 == 0:
                        print(f"Generated {len(corpus)} sentences...")
                    if len(corpus) >= num_sentences:
                        break

        # Adjust corpus to exact target size and distributions
        corpus = self._adjust_corpus_distribution(
            corpus,
            complexity_distribution,
            semantic_frame_distribution,
            target_size=num_sentences
        )

        # Store generation metadata
        metadata = {
            "generation_params": {
                "num_sentences": num_sentences,
                "complexity_distribution": complexity_distribution,
                "semantic_frame_distribution": semantic_frame_distribution,
                "template_weights": frames_weights,
                "include_annotations": include_annotations
            },
            "actual_distributions": self.calculate_actual_distributions(corpus),
            "lexicon_info": self.lexicon.export_lexicon_details(),
            "template_usage": self.frames.get_template_usage()
        }

        print(f"Corpus generation complete: {len(corpus)} sentences")
        return SynthCorpus(corpus, metadata)

    @staticmethod
    def calculate_frame_targets(num_sentences: int, frame_dist: Dict[str, float]) -> Dict[str, int]:
        """Calculate target counts for each semantic frame."""
        return {frame: int(num_sentences * ratio) for frame, ratio in frame_dist.items()}

    @staticmethod
    def calculate_complexity_targets(num_sentences: int, complexity_dist: Dict[str, float]) -> Dict[str, int]:
        """Calculate target counts for each complexity level."""
        return {complexity: int(num_sentences * ratio) for complexity, ratio in complexity_dist.items()}

    def _adjust_corpus_distribution(self,
                                    corpus: List[Dict[str, Any]],
                                    complexity_dist: Dict[str, float],
                                    frame_dist: Dict[str, float],
                                    target_size: int) -> List[Dict[str, Any]]:
        """Fine-tune corpus to match exact target distributions."""
        # Group by complexity and frame
        by_complexity = defaultdict(list)
        by_frame = defaultdict(list)

        for i, item in enumerate(corpus):
            if "metadata" in item:
                complexity = item["metadata"].get("complexity", "simple")
                frame = item["metadata"].get("frame", "transitive_action")
                by_complexity[complexity].append(i)
                by_frame[frame].append(i)

        # Adjust for target size
        if len(corpus) > target_size:
            excess = len(corpus) - target_size
            indices_to_remove = random.sample(range(len(corpus)), excess)
            corpus = [item for i, item in enumerate(corpus) if i not in indices_to_remove]
        elif len(corpus) < target_size:
            deficit = target_size - len(corpus)
            for _ in range(deficit):
                # Generate additional sentences with balanced distribution
                complexity = random.choices(
                    list(complexity_dist.keys()),
                    weights=list(complexity_dist.values())
                )[0]

                sentence_data = self.sentence_generator.generate_sentence(complexity=complexity)
                sentence_data["corpus_metadata"] = {
                    "sentence_id": len(corpus),
                    "adjustment_generation": True
                }
                corpus.append(sentence_data)

        return corpus

    @staticmethod
    def calculate_actual_distributions(corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate actual distributions in the generated corpus."""
        total = len(corpus)

        complexity_counts = Counter()
        frame_counts = Counter()

        for item in corpus:
            if "metadata" in item:
                complexity_counts[item["metadata"].get("complexity", "unknown")] += 1
                frame_counts[item["metadata"].get("frame", "unknown")] += 1

        return {
            "complexity": {k: v / total for k, v in complexity_counts.items()},
            "semantic_frames": {k: v / total for k, v in frame_counts.items()}
        }

    def evaluate_corpus(self, corpus: SynthCorpus, calculate_suitability: Optional[bool]=None) -> Dict[str, Any]:
        """
        Comprehensive corpus evaluation with semantic frame analysis.

        Returns:
            Dictionary with evaluation metrics and recommendations
        """
        # Standard corpus evaluation
        stats = self.evaluator.analyze_corpus(corpus.corpus_data())
        if calculate_suitability:
            suitability = self.evaluator.evaluate_next_token_prediction_suitability()

            print(f"Corpus evaluation complete:")
            print(f"  Suitability score: {suitability['suitability_score']:.2f} ({suitability['suitability_category']})")

            if suitability["recommendations"]:
                print("\nRecommendations:")
                for i, rec in enumerate(suitability["recommendations"], 1):
                    print(f"  {i}. {rec}")

            return {
                "statistics": stats,
                "suitability": suitability,
                "metadata": corpus.metadata
            }
        else:

            return {
                "statistics": stats,
                "metadata": corpus.metadata
            }