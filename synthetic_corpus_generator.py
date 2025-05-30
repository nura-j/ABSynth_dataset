from lexicon_generator import LexiconGenerator
from template_manager import TemplateManager
from sentence_generator import SentenceGenerator
from corpus_evaluator import CorpusEvaluator
import json
import os
import argparse
import matplotlib.pyplot as plt

class SyntheticCorpusGenerator:
    """
    Main class for generating synthetic corpora with controlled statistical
    properties for next token prediction and other NLP tasks.
    """
    
    def __init__(self, vocab_sizes=None):
        """
        Initialize the corpus generator with default or custom vocabulary sizes.
        
        Args:
            vocab_sizes: Optional dictionary mapping word categories to vocab sizes
        """
        # Default vocabulary sizes if not provided
        self.vocab_sizes = vocab_sizes or {
            "noun": 200,
            "transitive_verb": 30,
            "intransitive_verb": 30,
            "communication_verb": 15,
            "motion_verb": 15,
            "change_verb": 15,
            "adjective": 25,
            "adverb": 15,
            "location": 100,
            "temporal": 20,
        }
        
        # Initialize components
        self.lexicon = LexiconGenerator(self.vocab_sizes)
        self.templates = TemplateManager()
        self.sentence_generator = SentenceGenerator(self.lexicon, self.templates)
        self.evaluator = CorpusEvaluator()
        self.corpus = []
    
    def generate_corpus(self, num_sentences, complexity_distribution=None):
        """
        Generate a corpus with specified number of sentences and complexity distribution.
        
        Args:
            num_sentences: Number of sentences to generate
            complexity_distribution: Optional dict with target distribution
                e.g., {"simple": 0.55, "medium": 0.35, "complex": 0.1}
        
        Returns:
            List of generated sentence dictionaries
        """
        # Default complexity distribution if not provided
        complexity_distribution = complexity_distribution or {
            "simple": 0.55, "medium": 0.35, "complex": 0.1
        }
        
        self.corpus = []
        
        # Generate sentences
        print(f"Generating {num_sentences} sentences...")
        for i in range(num_sentences):
            if i % 1000 == 0 and i > 0:
                print(f"Generated {i} sentences...")
            
            sentence = self.sentence_generator.generate_sentence()
            self.corpus.append(sentence)
        
        # Adjust corpus to match target distribution
        self._adjust_corpus_distribution(complexity_distribution, num_sentences)
        
        # Update evaluator with the generated corpus
        self.evaluator.corpus = self.corpus
        
        return self.corpus
    
    def _adjust_corpus_distribution(self, target_dist, num_sentences):
        """
        Adjust corpus to match target complexity distribution.
        
        Args:
            target_dist: Dictionary with target distribution percentages
            num_sentences: Target number of sentences
        """
        import random
        from collections import defaultdict
        
        # Group sentences by complexity
        by_complexity = defaultdict(list)
        for i, item in enumerate(self.corpus):
            complexity = item["metadata"]["complexity"]
            by_complexity[complexity].append(i)
        
        # Calculate current counts and target counts
        current_counts = {k: len(v) for k, v in by_complexity.items()}
        target_counts = {k: int(v * num_sentences) for k, v in target_dist.items()}
        
        # Add sentences for complexities that need more
        for complexity, target in target_counts.items():
            current = current_counts.get(complexity, 0)
            
            if current < target:
                # Generate additional sentences of this complexity
                for _ in range(target - current):
                    sentence = self.sentence_generator.generate_sentence(complexity)
                    self.corpus.append(sentence)
        
        # Trim corpus if too large
        if len(self.corpus) > num_sentences:
            # Recalculate groups
            by_complexity = defaultdict(list)
            for i, item in enumerate(self.corpus):
                complexity = item["metadata"]["complexity"]
                by_complexity[complexity].append(i)
            
            current_counts = {k: len(v) for k, v in by_complexity.items()}
            
            # Calculate how many to remove from each group
            to_remove = {}
            for complexity in by_complexity:
                current = current_counts.get(complexity, 0)
                target = target_counts.get(complexity, 0)
                to_remove[complexity] = max(0, current - target)
            
            # Create list of indices to remove
            indices_to_remove = []
            for complexity, count in to_remove.items():
                indices = by_complexity[complexity]
                # Randomly select indices to remove
                if count > 0 and indices:
                    selected = random.sample(indices, min(count, len(indices)))
                    indices_to_remove.extend(selected)
            
            # Create new corpus without removed indices
            self.corpus = [item for i, item in enumerate(self.corpus) if i not in indices_to_remove]
    
    def save_corpus(self, output_file, include_stats=False):
        """
        Save the generated corpus to a JSON file.
        
        Args:
            output_file: Output filename
            include_stats: Whether to include corpus statistics
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")
        
        output = {"corpus": self.corpus}

        if include_stats:
            output["statistics"] = self.evaluator.analyze_corpus()
        else:
            # Remove statistics from output
            for item in self.corpus:
                item.pop("metadata", None)
            output = output["corpus"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Corpus with {len(self.corpus)} sentences saved to '{output_file}'")
    
    def evaluate_corpus(self):
        """
        Evaluate the generated corpus for next token prediction suitability.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")
        
        # Analyze corpus statistics
        self.evaluator.corpus = self.corpus
        stats = self.evaluator.analyze_corpus()
        
        # Evaluate suitability for next token prediction
        suitability = self.evaluator.evaluate_next_token_prediction_suitability()
        
        print(f"Corpus evaluation complete. Suitability score: {suitability['suitability_score']:.2f} ({suitability['suitability_category']})")
        
        if suitability["recommendations"]:
            print("\nRecommendations for improvement:")
            for i, rec in enumerate(suitability["recommendations"], 1):
                print(f"{i}. {rec}")
        
        return {
            "statistics": stats,
            "suitability": suitability
        }
    
    def plot_statistics(self, output_dir=None):
        """
        Generate plots of corpus statistics.
        
        Args:
            output_dir: Optional directory to save plots
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")
        
        self.evaluator.corpus = self.corpus
        self.evaluator.plot_statistics(output_dir)
    
    def generate_evaluation_report(self, output_file):
        """
        Generate and save a comprehensive evaluation report.
        
        Args:
            output_file: Output filename for the report
        
        Returns:
            Dictionary with evaluation report
        """
        if not self.corpus:
            raise ValueError("No corpus generated yet. Call generate_corpus() first.")
        
        self.evaluator.corpus = self.corpus
        report = self.evaluator.generate_report(output_file)
        
        return report


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