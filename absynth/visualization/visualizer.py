from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from absynth import SyntheticCorpusGenerator


class Visualizer:
    def __init__(self, log_dir: str = None, save_visualization: bool = True, config=None):
        self.config = config
        self.log_dir = log_dir
        self.save_visualization = save_visualization
        plt.style.use('seaborn-v0_8-whitegrid')
        if log_dir:
            import os
            os.makedirs(log_dir, exist_ok=True)

    def visualize(self, corpus):
        """
        We expect the stats to be a dictionary with keys like 'token_count', 'vocab_size', etc.
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
        :param corpus:
        :return:
        """
        self.stats = corpus.statistics
        if not self.stats:
            from absynth.corpus.corpus_evaluator import CorpusEvaluator
            generator = SyntheticCorpusGenerator()
            self.evalulation = generator.evaluate_corpus(corpus, calculate_suitability=True)
            self.stats = self.evalulation["statistics"]
        self._visualize_token_statistics()
        self._visualize_sentence_statistics()
        self._visualize_complexity_distribution()
        self._visualize_semantic_frame_analysis()

    def _visualize_token_statistics(self):

        token_stats = self.stats.get("token_statistics", {})
        if not token_stats:
            print("No token statistics available for visualization.")
            return

        total_words = token_stats.get("total_words", 0)
        unique_words = token_stats.get("unique_words", 0)
        type_token_ratio = token_stats.get("type_token_ratio", 0.0)
        top_words = token_stats.get("top_words", [])

        # Create a bar chart for top words
        if top_words:
            words, counts = zip(*top_words)
            plt.bar(words, counts)
            plt.title('Top Words in Corpus')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/top_words.png")
            plt.show()

        print(f"Total Words: {total_words}, Unique Words: {unique_words}, Type-Token Ratio: {type_token_ratio:.2f}")


    def _visualize_sentence_statistics(self):
        sentence_stats = self.stats.get("sentence_statistics", {})
        if not sentence_stats:
            print("No sentence statistics available for visualization.")
            return

        avg_length = sentence_stats.get("avg_length", 0.0)
        length_std = sentence_stats.get("length_std", 0.0)
        min_length = sentence_stats.get("min_length", 0)
        max_length = sentence_stats.get("max_length", 0)
        length_distribution = sentence_stats.get("length_distribution", {})

        # Create a histogram for sentence lengths
        if length_distribution:
            lengths, counts = zip(*length_distribution.items())
            plt.bar(lengths, counts)
            plt.title('Sentence Length Distribution')
            plt.xlabel('Sentence Length')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/sentence_length_distribution.png")
            plt.show()

        print(f"Avg Length: {avg_length:.2f}, Length Std: {length_std:.2f}, Min Length: {min_length}, Max Length: {max_length}")


    def _visualize_complexity_distribution(self):
        complexity_dist = self.stats.get("complexity_distribution", {})
        if not complexity_dist:
            print("No complexity distribution available for visualization.")
            return

        # Assuming complexity_dist contains a list of (complexity_level, frequency) tuples
        complexities = list(complexity_dist.keys())
        frequencies = list(complexity_dist.values())

        if complexities and frequencies:
            plt.bar(complexities, frequencies)
            plt.title('Complexity Distribution')
            plt.xlabel('Complexity Level')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/complexity_distribution.png")
            plt.show()

    def _visualize_semantic_frame_analysis(self):
        semantic_analysis = self.stats.get("semantic_frame_analysis", {})
        if not semantic_analysis:
            print("No semantic frame analysis available for visualization.")
            return

        frame_distribution = semantic_analysis.get("frame_distribution", {})
        role_distribution = semantic_analysis.get("role_distribution", {})
        avg_arguments_per_sentence = semantic_analysis.get("avg_arguments_per_sentence", 0.0)

        if frame_distribution:
            frames, counts = zip(*frame_distribution.items())
            plt.bar(frames, counts)
            plt.title('Semantic Frame Distribution')
            plt.xlabel('Frames')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/semantic_frame_distribution.png")
            plt.show()
        if role_distribution:
            roles, counts = zip(*role_distribution.items())
            plt.bar(roles, counts)
            plt.title('Semantic Role Distribution')
            plt.xlabel('Roles')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/semantic_role_distribution.png")
            plt.show()
        print(f"Avg Arguments per Sentence: {avg_arguments_per_sentence:.2f}")
