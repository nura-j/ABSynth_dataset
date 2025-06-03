import numpy as np
import matplotlib.pyplot as plt
import json
import math
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple
from ..lexicon.semantic_roles import SemanticRoles


class CorpusEvaluator:
    """
    Evaluates synthetic corpus with focus on semantic frame diversity
    and next token prediction suitability.
    """

    def __init__(self):
        """Initialize corpus evaluator."""
        self.statistics = {}

    def analyze_corpus(self, corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive corpus analysis including semantic frame metrics.

        Args:
            corpus: List of sentence dictionaries

        Returns:
            Dictionary with detailed corpus statistics
        """
        # Extract basic components
        sentences = [item["sentence"] for item in corpus]
        all_words = []

        print(f"Analyzing {len(sentences)} sentences...")
        for sentence in sentences:
            all_words.extend(sentence.lower().split())

        # Basic statistics
        word_counts = Counter(all_words)
        total_words = len(all_words)
        unique_words = len(word_counts)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0

        # Sentence length analysis
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        length_std = np.std(lengths) if lengths else 0

        # Zipfian analysis
        zipf_analysis = self._analyze_zipfian_distribution(word_counts)

        # Entropy analysis
        entropy_analysis = self._analyze_entropy(corpus)

        # Complexity distribution
        complexity_dist = self._analyze_complexity_distribution(corpus)

        # Semantic frame analysis
        semantic_analysis = self._analyze_semantic_frames(corpus)

        # Bigram statistics
        bigram_stats = self._analyze_bigrams(all_words)

        self.statistics = {
            "corpus_size": len(corpus),
            "token_statistics": {
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": type_token_ratio,
                "top_words": word_counts.most_common(20)
            },
            "sentence_statistics": {
                "avg_length": avg_length,
                "length_std": length_std,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "length_distribution": self._get_length_distribution(lengths)
            },
            "zipfian_analysis": zipf_analysis,
            "entropy_statistics": entropy_analysis,
            "complexity_distribution": complexity_dist,
            "semantic_frame_analysis": semantic_analysis,
            "bigram_statistics": bigram_stats
        }

        return self.statistics

    def _analyze_zipfian_distribution(self, word_counts: Counter) -> Dict[str, float]:
        """Analyze how well the corpus follows Zipf's law.
        Zipf's law => frequency ∝ 1/rank^α, where α ≈ 1 for natural language.
        So for log-log space: log(frequency) = log(C) - α * log(rank)
        code inspired by: https://github.com/chasmani/zipfanalysis/tree/master
        """
        if len(word_counts) < 2:
            return {"zipf_coefficient": 0, "zipf_quality": 0}

        ranks = np.arange(1, len(word_counts) + 1)
        frequencies = [count for _, count in word_counts.most_common()]

        log_ranks = np.log(ranks)
        log_frequencies = np.log(np.array(frequencies))

        # Linear regression on log-log scale
        slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
        zipf_coefficient = slope
        zipf_quality = 1.0 - abs(zipf_coefficient + 1.0)  # Closer to 1.0 is better meaning it follows Zipf's law

        return {
            "zipf_coefficient": zipf_coefficient,
            "ideal_coefficient": -1.0,
            "zipf_quality": max(0, zipf_quality)
        }

    def _analyze_entropy(self, corpus: List[Dict], thresholds: Optional[Tuple[float, float]] = None) -> Dict[
        str, float]:
        """Analyze entropy distribution for next token prediction."""
        entropy_values = []

        for item in corpus:
            metadata = item.get("metadata", {})
            if "entropy_profile" in metadata:
                entropy_values.extend(metadata["entropy_profile"])

        if not entropy_values:
            return {
                "avg_entropy": 0,
                "entropy_std": 0,
                "predictability_distribution": {
                    "high_predictability": 0,
                    "medium_predictability": 0,
                    "low_predictability": 0
                }
            }

        avg_entropy = sum(entropy_values) / len(entropy_values)
        entropy_std = np.std(entropy_values)
        total = len(entropy_values)

        if thresholds:
            high_thresh, low_thresh = thresholds
        else:
            # Adaptive thresholds based on quantiles
            sorted_entropy = sorted(entropy_values)
            high_thresh = sorted_entropy[int(0.33 * total)]
            low_thresh = sorted_entropy[int(0.66 * total)]

        high_pred = sum(1 for e in entropy_values if e < high_thresh) / total
        medium_pred = sum(1 for e in entropy_values if high_thresh <= e < low_thresh) / total
        low_pred = sum(1 for e in entropy_values if e >= low_thresh) / total

        return {
            "avg_entropy": avg_entropy,
            "entropy_std": entropy_std,
            "min_entropy": min(entropy_values),
            "max_entropy": max(entropy_values),
            "predictability_distribution": {
                "high_predictability": high_pred,
                "medium_predictability": medium_pred,
                "low_predictability": low_pred
            }
        }

    def _analyze_complexity_distribution(self, corpus: List[Dict]) -> Dict[str, float]:
        """Analyze sentence complexity distribution."""
        complexities = Counter()

        for item in corpus:
            complexity = item.get("metadata", {}).get("complexity", "unknown")
            complexities[complexity] += 1

        total = sum(complexities.values())
        return {k: v / total for k, v in complexities.items()} if total > 0 else {}

    def _analyze_semantic_frames(self, corpus: List[Dict]) -> Dict[str, Any]:
        """Analyze semantic frame distribution and role usage."""
        frame_counts = Counter()
        role_counts = Counter()
        argument_counts = Counter()

        for item in corpus:
            # Frame analysis
            frame = item.get("metadata", {}).get("frame", "unknown")
            frame_counts[frame] += 1

            # Role analysis
            semantic_roles = item.get("semantic_roles", {})
            for role_info in semantic_roles.values():
                role = role_info.get("role", "unknown")
                role_counts[role] += 1

            # Argument structure analysis
            num_args = len(semantic_roles)
            argument_counts[num_args] += 1

        total_sentences = len(corpus)

        return {
            "frame_distribution": {k: v / total_sentences for k, v in frame_counts.items()},
            "role_distribution": dict(role_counts),
            "argument_structure_distribution": {k: v / total_sentences for k, v in argument_counts.items()},
            "unique_frames": len(frame_counts),
            "unique_roles": len(role_counts),
            "avg_arguments_per_sentence": sum(
                k * v for k, v in argument_counts.items()) / total_sentences if total_sentences > 0 else 0
        }

    def _analyze_bigrams(self, all_words: List[str]) -> Dict[str, Any]:
        """Analyze bigram statistics."""
        bigrams = [(all_words[i], all_words[i + 1]) for i in range(len(all_words) - 1)]
        bigram_counts = Counter(bigrams)

        unique_bigrams = len(bigram_counts)
        total_bigrams = len(bigrams)
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0

        return {
            "unique_bigrams": unique_bigrams,
            "total_bigrams": total_bigrams,
            "bigram_diversity": bigram_diversity,
            "top_bigrams": bigram_counts.most_common(10)
        }

    def _get_length_distribution(self, lengths: List[int]) -> Dict[int, int]:
        """Get distribution of sentence lengths."""
        return dict(Counter(lengths))

    def evaluate_next_token_prediction_suitability(self, ideal_dist: Optional[Dict[str, float]]=None, num_standard_roles: Optional[int]=None) -> Dict[str, Any]:
        """
        Evaluate corpus suitability for next token prediction with semantic awareness.

        Returns:
            Dictionary with suitability metrics and recommendations
        """
        if not self.statistics:
            raise ValueError("No statistics available. Run analyze_corpus() first.")

        stats = self.statistics

        # Extract key metrics
        zipf_quality = stats["zipfian_analysis"]["zipf_quality"]
        entropy_std = stats["entropy_statistics"]["entropy_std"]
        type_token_ratio = stats["token_statistics"]["type_token_ratio"]

        # Semantic metrics
        semantic_stats = stats["semantic_frame_analysis"]
        frame_diversity = semantic_stats["unique_frames"] / max(1, stats["corpus_size"])
        if num_standard_roles is None:
            num_standard_roles = SemanticRoles.get_standard_roles_count()
        role_coverage = min(1.0, semantic_stats["unique_roles"] / num_standard_roles)

        # Predictability balance
        pred_dist = stats["entropy_statistics"]["predictability_distribution"]
        if ideal_dist is None:
            ideal_dist = {"high_predictability": 0.3, "medium_predictability": 0.5, "low_predictability": 0.2}
        pred_balance = 1.0 - sum(abs(pred_dist.get(k, 0) - v) for k, v in ideal_dist.items()) / 2

        # Overall suitability score (0-1)
        suitability_score = (
                zipf_quality * 0.20 +
                min(1.0, entropy_std) * 0.20 +
                (1.0 - abs(type_token_ratio - 0.1) * 5) * 0.20 +
                pred_balance * 0.20 +
                frame_diversity * 0.10 +
                role_coverage * 0.10
        )

        # Categorize suitability
        if suitability_score > 0.8:
            category = "excellent"
        elif suitability_score > 0.6:
            category = "good"
        elif suitability_score > 0.4:
            category = "adequate"
        else:
            category = "needs improvement"

        # Generate recommendations
        recommendations = self._generate_recommendations(stats)

        return {
            "suitability_score": suitability_score,
            "suitability_category": category,
            "component_scores": {
                "zipf_quality": zipf_quality,
                "entropy_variation": min(1.0, entropy_std),
                "vocabulary_balance": max(0, 1.0 - abs(type_token_ratio - 0.1) * 5),
                "predictability_balance": pred_balance,
                "semantic_diversity": frame_diversity,
                "role_coverage": role_coverage
            },
            "recommendations": recommendations
        }

    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate specific recommendations for corpus improvement."""
        recommendations = []

        # Zipf analysis
        zipf_quality = stats["zipfian_analysis"]["zipf_quality"]
        if zipf_quality < 0.7:
            recommendations.append(
                f"Improve word frequency distribution (Zipf quality: {zipf_quality:.2f}, target: >0.7)"
            )

        # Entropy variation
        entropy_std = stats["entropy_statistics"]["entropy_std"]
        if entropy_std < 0.5:
            recommendations.append(
                f"Increase entropy variation for diverse prediction challenges (std: {entropy_std:.2f}, target: >0.5)"
            )

        # Semantic diversity
        semantic_stats = stats["semantic_frame_analysis"]
        if semantic_stats["unique_frames"] < 4:
            recommendations.append(
                f"Increase semantic frame diversity (current: {semantic_stats['unique_frames']}, target: ≥4 frames)"
            )

        # Role coverage
        role_coverage = semantic_stats["unique_roles"] / max(1, SemanticRoles.get_standard_roles_count())
        if role_coverage < 0.8:
            recommendations.append(
                f"Improve semantic role coverage (current: {role_coverage:.1%}, target: ≥80%)"
            )

        return recommendations