import numpy as np
import matplotlib.pyplot as plt
import json
import math
from collections import Counter

class CorpusEvaluator:
    """
    Evaluates a synthetic corpus for next token prediction and other NLP tasks.
    Provides detailed metrics and visualizations for quality assessment.
    """
    
    def __init__(self, corpus=None):
        """
        Initialize with optional corpus data.
        
        Args:
            corpus: Optional list of sentence dictionaries
        """
        self.corpus = corpus or []
        self.statistics = {}
    
    def load_corpus(self, corpus_file):
        """
        Load corpus from a JSON file.
        
        Args:
            corpus_file: Path to corpus JSON file
        """
        with open(corpus_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.corpus = data
        elif 'corpus' in data:
            self.corpus = data['corpus']
        else:
            raise ValueError("Invalid corpus format")
        
        print(f"Loaded {len(self.corpus)} sentences")
    
    def analyze_corpus(self):
        """
        Perform comprehensive analysis of corpus properties.
        
        Returns:
            Dictionary with corpus statistics
        """
        # Extract basic components
        sentences = [item["sentence"] for item in self.corpus]
        all_words = []
        print(f"Analyzing {len(sentences)} sentences...")
        for sentence in sentences:
            all_words.extend(sentence.lower().split())
        
        word_counts = Counter(all_words)
        total_words = len(all_words)
        unique_words = len(word_counts)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0
        
        # Extract sentence lengths
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        length_std = (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5 if lengths else 0
        
        # Zipfian analysis
        ranks = np.arange(1, len(word_counts) + 1)
        frequencies = [count for _, count in word_counts.most_common()]
        
        if len(frequencies) > 1:
            log_ranks = np.log(ranks)
            log_frequencies = np.log(np.array(frequencies))
            
            # Linear regression on log-log scale
            slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
            zipf_coefficient = slope
            zipf_quality = 1.0 - abs(zipf_coefficient + 1.0)  # Closer to 1.0 is better (ideal slope is -1.0)
        else:
            zipf_coefficient = 0
            zipf_quality = 0
        
        # Entropy analysis
        entropy_values = []
        complexities = Counter()
        
        for item in self.corpus:
            if "metadata" in item:
                if "entropy_profile" in item["metadata"]:
                    entropy_values.extend(item["metadata"]["entropy_profile"])
                if "complexity" in item["metadata"]:
                    complexities[item["metadata"]["complexity"]] += 1
        
        avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0
        entropy_std = np.std(entropy_values) if entropy_values else 0
        
        # Calculate distribution of entropy values (predictability)
        if entropy_values:
            high_predictability = sum(1 for e in entropy_values if e < 1.5) / len(entropy_values)
            medium_predictability = sum(1 for e in entropy_values if 1.5 <= e < 3.0) / len(entropy_values)
            low_predictability = sum(1 for e in entropy_values if e >= 3.0) / len(entropy_values)
            predictability_dist = {
                "high_predictability": high_predictability,
                "medium_predictability": medium_predictability,
                "low_predictability": low_predictability
            }
        else:
            predictability_dist = {"high_predictability": 0, "medium_predictability": 0, "low_predictability": 0}
        
        # Complexity distribution
        total_sentences = len(self.corpus)
        complexity_dist = {k: v/total_sentences for k, v in complexities.items()} if total_sentences > 0 else {}
        
        # Calculate bigram statistics
        bigrams = []
        for i in range(len(all_words) - 1):
            bigrams.append((all_words[i], all_words[i+1]))
        
        bigram_counts = Counter(bigrams)
        unique_bigrams = len(bigram_counts)
        bigram_diversity = unique_bigrams / len(bigrams) if bigrams else 0
        
        # Store all statistics
        self.statistics = {
            "corpus_size": len(self.corpus),
            "token_statistics": {
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": type_token_ratio,
                "top_words": [(w, c) for w, c in word_counts.most_common(20)]

            },
            "sentence_statistics": {
                "avg_length": avg_length,
                "length_std": length_std,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0
            },
            "zipfian_analysis": {
                "zipf_coefficient": zipf_coefficient,
                "ideal_coefficient": -1.0,
                "zipf_quality": zipf_quality
            },
            "entropy_statistics": {
                "avg_entropy": avg_entropy,
                "entropy_std": entropy_std,
                "predictability_distribution": predictability_dist
            },
            "complexity_distribution": complexity_dist,
            "bigram_statistics": {
                "unique_bigrams": unique_bigrams,
                "total_bigrams": len(bigrams),
                "bigram_diversity": bigram_diversity
            }
        }
        
        return self.statistics
    
    def evaluate_next_token_prediction_suitability(self):
        """
        Evaluate how suitable this corpus is for next token prediction tasks.
        
        Returns:
            Dictionary with evaluation metrics and recommendations
        """
        if not self.statistics:
            self.analyze_corpus()
        
        stats = self.statistics
        
        # Check key metrics for next token prediction suitability
        zipf_quality = stats["zipfian_analysis"]["zipf_quality"]
        entropy_std = stats["entropy_statistics"]["entropy_std"]
        type_token_ratio = stats["token_statistics"]["type_token_ratio"]
        complex_prop = stats["complexity_distribution"].get("complex", 0)
        
        # Calculate predictability balance score
        pred_dist = stats["entropy_statistics"]["predictability_distribution"]
        ideal_dist = {"high_predictability": 0.3, "medium_predictability": 0.5, "low_predictability": 0.2}
        pred_balance = 1.0 - sum(abs(pred_dist.get(k, 0) - v) for k, v in ideal_dist.items()) / 2
        
        # Overall suitability score (0-1)
        suitability_score = (
            zipf_quality * 0.25 +
            min(1.0, entropy_std) * 0.25 +
            (1.0 - abs(type_token_ratio - 0.1) * 5) * 0.25 +
            pred_balance * 0.25
        )
        
        # Categorize suitability
        if suitability_score > 0.8:
            suitability_category = "excellent"
        elif suitability_score > 0.6:
            suitability_category = "good"
        elif suitability_score > 0.4:
            suitability_category = "adequate"
        else:
            suitability_category = "needs improvement"
        
        # Generate specific recommendations
        recommendations = []
        
        if zipf_quality < 0.7:
            recommendations.append(
                f"Improve word frequency distribution to better match Zipf's law "
                f"(current coefficient: {stats['zipfian_analysis']['zipf_coefficient']:.2f}, "
                "target: -1.0)"
            )
        
        if entropy_std < 0.5:
            recommendations.append(
                f"Increase entropy variation for more diverse prediction challenges "
                f"(current std: {entropy_std:.2f}, target: >0.5)"
            )
        
        if abs(type_token_ratio - 0.1) > 0.05:
            if type_token_ratio < 0.1:
                recommendations.append(
                    f"Increase vocabulary diversity (current type-token ratio: {type_token_ratio:.3f}, "
                    "target: 0.1-0.15)"
                )
            else:
                recommendations.append(
                    f"Reduce vocabulary diversity for better pattern learning "
                    f"(current type-token ratio: {type_token_ratio:.3f}, target: 0.1-0.15)"
                )
        
        if complex_prop < 0.1:
            recommendations.append(
                f"Increase proportion of complex sentences with long-distance dependencies "
                f"(current: {complex_prop:.1%}, target: 10%)"
            )
        
        if pred_dist.get("medium_predictability", 0) < 0.4:
            recommendations.append(
                f"Adjust templates to increase proportion of medium-predictability contexts "
                f"(current: {pred_dist.get('medium_predictability', 0):.1%}, target: 50%)"
            )
        
        return {
            "suitability_score": suitability_score,
            "suitability_category": suitability_category,
            "component_scores": {
                "zipf_quality": zipf_quality,
                "entropy_variation": min(1.0, entropy_std),
                "vocabulary_balance": (1.0 - abs(type_token_ratio - 0.1) * 5),
                "predictability_balance": pred_balance
            },
            "recommendations": recommendations
        }
    
    def plot_statistics(self, output_dir=None):
        """
        Generate plots of corpus statistics.
        
        Args:
            output_dir: Optional directory to save plots
        """
        if not self.statistics:
            self.analyze_corpus()
        
        stats = self.statistics
        
        # 1. Plot word frequency distribution (Zipfian)
        plt.figure(figsize=(10, 6))
        word_counts = [count for _, count in stats["token_statistics"]["top_words"]]
        ranks = np.arange(1, len(word_counts) + 1)
        
        if word_counts:
            plt.loglog(ranks, word_counts, 'bo-', alpha=0.6, label='Observed')
            
            # Plot ideal Zipfian line for comparison
            ideal_counts = [word_counts[0] * (1/i) for i in ranks]
            plt.loglog(ranks, ideal_counts, 'r--', label='Ideal Zipfian')
            
            plt.title(f'Word Frequency Distribution (Zipf coefficient: {stats["zipfian_analysis"]["zipf_coefficient"]:.2f})')
            plt.xlabel('Rank')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(f"{output_dir}/zipf_distribution.png", dpi=300, bbox_inches='tight')
        
        # 2. Plot sentence length distribution
        plt.figure(figsize=(10, 6))
        lengths = [item.get("metadata", {}).get("length", 0) for item in self.corpus]
        if lengths:
            plt.hist(lengths, bins=15, alpha=0.7)
            plt.axvline(stats["sentence_statistics"]["avg_length"], color='r', linestyle='--', 
                      label=f'Mean: {stats["sentence_statistics"]["avg_length"]:.1f}')
            
            plt.title('Sentence Length Distribution')
            plt.xlabel('Number of Words')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(f"{output_dir}/sentence_length.png", dpi=300, bbox_inches='tight')
        
        # 3. Plot entropy distribution
        plt.figure(figsize=(10, 6))
        entropy_values = []
        for item in self.corpus:
            if "metadata" in item and "entropy_profile" in item["metadata"]:
                entropy_values.extend(item["metadata"]["entropy_profile"])
        
        if entropy_values:
            plt.hist(entropy_values, bins=20, alpha=0.7, range=(0, 5))
            plt.axvline(stats["entropy_statistics"]["avg_entropy"], color='r', linestyle='--',
                      label=f'Mean: {stats["entropy_statistics"]["avg_entropy"]:.2f}')
            
            plt.title('Distribution of Token Entropy (Predictability)')
            plt.xlabel('Entropy (bits)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(f"{output_dir}/entropy_distribution.png", dpi=300, bbox_inches='tight')
        
        # 4. Plot complexity distribution
        plt.figure(figsize=(10, 6))
        complexities = stats["complexity_distribution"]
        if complexities:
            plt.bar(complexities.keys(), complexities.values(), alpha=0.7)
            plt.title('Sentence Complexity Distribution')
            plt.xlabel('Complexity Level')
            plt.ylabel('Proportion')
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(f"{output_dir}/complexity_distribution.png", dpi=300, bbox_inches='tight')
        
        # 5. Plot predictability distribution
        plt.figure(figsize=(10, 6))
        pred_dist = stats["entropy_statistics"]["predictability_distribution"]
        if pred_dist:
            plt.bar(pred_dist.keys(), pred_dist.values(), alpha=0.7)
            plt.title('Token Predictability Distribution')
            plt.xlabel('Predictability Level')
            plt.ylabel('Proportion')
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(f"{output_dir}/predictability_distribution.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, output_file=None):
        """
        Generate comprehensive evaluation report with recommendations.
        
        Args:
            output_file: Optional file to save report JSON
        
        Returns:
            Dictionary with evaluation report
        """
        stats = self.analyze_corpus()
        suitability = self.evaluate_next_token_prediction_suitability()
        
        report = {
            "corpus_statistics": stats,
            "next_token_prediction_suitability": suitability,
            "summary": {
                "corpus_size": stats["corpus_size"],
                "type_token_ratio": stats["token_statistics"]["type_token_ratio"],
                "zipf_coefficient": stats["zipfian_analysis"]["zipf_coefficient"],
                "avg_entropy": stats["entropy_statistics"]["avg_entropy"],
                "entropy_std": stats["entropy_statistics"]["entropy_std"],
                "suitability_score": suitability["suitability_score"],
                "suitability_category": suitability["suitability_category"]
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Evaluation report saved to {output_file}")
        
        return report