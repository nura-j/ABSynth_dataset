import matplotlib.pyplot as plt
from absynth import SyntheticCorpusGenerator
import seaborn as sns


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

        if top_words:
            words, counts = zip(*top_words)

            # Horizontal bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts, y=words, orient='h')
            plt.title('Top Words in Corpus')
            plt.xlabel('Frequency')
            plt.ylabel('Words')
            plt.tight_layout()
            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/top_words_horizontal.png", dpi=300)
            plt.show()

            # Cumulative distribution plot
            cumulative = [sum(counts[:i + 1]) / sum(counts) for i in range(len(counts))]
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o')
            plt.title('Cumulative Frequency of Top Words')
            plt.xlabel('Rank')
            plt.ylabel('Cumulative Frequency')
            plt.grid(True)
            plt.tight_layout()
            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/top_words_cumulative.png", dpi=300)
            plt.show()

        print(f"Total Words: {total_words}, Unique Words: {unique_words}, Type-Token Ratio: {type_token_ratio:.2f}")



    def _visualize_sentence_statistics(self):
        sentence_stats = self.stats.get("sentence_statistics", {})
        if not sentence_stats:
            print("No sentence statistics available for visualization.")
            return

        length_distribution = sentence_stats.get("length_distribution", {})
        if not length_distribution:
            print("No length distribution data.")
            return

        lengths, counts = zip(*length_distribution.items())
        data = [length for length, count in zip(lengths, counts) for _ in range(count)]

        sns.histplot(data, bins=20, kde=True)
        plt.title("Sentence Length Distribution")
        plt.xlabel("Sentence Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        if self.save_visualization and self.log_dir:
            plt.savefig(f"{self.log_dir}/sentence_length_hist_kde.png", dpi=300)
        plt.show()

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
                plt.savefig(f"{self.log_dir}/complexity_distribution.png", dpi=300)
            plt.show()

    import seaborn as sns

    def _visualize_semantic_frame_analysis(self):
        semantic_analysis = self.stats.get("semantic_frame_analysis", {})
        if not semantic_analysis:
            print("No semantic frame analysis available for visualization.")
            return

        frame_distribution = semantic_analysis.get("frame_distribution", {})
        role_distribution = semantic_analysis.get("role_distribution", {})
        avg_arguments_per_sentence = semantic_analysis.get("avg_arguments_per_sentence", 0.0)

        # Frame Distribution – horizontal bar plot
        if frame_distribution:
            frames, counts = zip(*sorted(frame_distribution.items(), key=lambda x: x[1], reverse=True))
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts, y=frames, orient='h')
            plt.title('Semantic Frame Distribution')
            plt.xlabel('Frequency')
            plt.ylabel('Frames')
            plt.tight_layout()
            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/semantic_frame_distribution.png", dpi=300)
            plt.show()

        # Role Distribution – horizontal bar plot
        if role_distribution:
            roles, counts = zip(*sorted(role_distribution.items(), key=lambda x: x[1], reverse=True))
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts, y=roles, orient='h')
            plt.title('Semantic Role Distribution')
            plt.xlabel('Frequency')
            plt.ylabel('Roles')
            plt.tight_layout()
            if self.save_visualization and self.log_dir:
                plt.savefig(f"{self.log_dir}/semantic_role_distribution.png", dpi=300)
            plt.show()

        print(f"Avg Arguments per Sentence: {avg_arguments_per_sentence:.2f}")
