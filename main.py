import argparse
import json

from absynth.corpus.synthetic_corpus_generator import SyntheticCorpusGenerator
import os
from absynth import Vocabulary

# Create output directories
os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)

# # Step 1: Define vocabulary sizes
# vocab_sizes = {
#     'noun': 2783,##
#      'transitive_verb': 694, #
#      'intransitive_verb': 694,#
#      'communication_verb': 347,#
#      'motion_verb': 347,#
#       'change_verb': 347,#
#      'adjective': 1388, ##
#      'adverb': 555, ##
#      'location': 694, ##
#      'temporal': 694, ##
#      'preposition': 416, ##
#      'determiner': 111, ##
#      'conjunction': 277, ##,
#      'result': 277, ##
#      # 'comp': 277,##
#      # 'rel': 277,##
#     # 'result': 277,##
# }
# # Step 2: Create generator with custom vocabulary sizes
# generator = SyntheticCorpusGenerator(vocab_sizes)
#
# # Step 3: Generate XXX sentences with a specific complexity distribution
# # This distribution is optimized for next token prediction tasks
# complexity_distribution = {
#     "simple": 0.50,     # 55% simple sentences
#     "medium": 0.30,     # 35% medium complexity
#     "complex": 0.20     # 10% complex with long-distance dependencies
# }
# corpus_size = 25000  # Number of sentences to generate
# corpus = generator.generate_corpus(corpus_size, complexity_distribution)
# print(f"Generated {len(corpus)} sentences for original ABSynth corpus")
# generator.save_corpus('output/absynth_original.json', include_stats=False)
# corpus = generator.generate_corpus(25000, complexity_distribution)
# print(f"Generated {len(corpus)} sentences")
#
# # Step 4: Save the corpus with statistics
# generator.save_corpus('output/next_token_corpus_4.json', include_stats=True)
# # #
# # Step 5: Evaluate corpus for next token prediction suitability
# evaluation = generator.evaluate_corpus()
#
# # Step 6: Generate plots to visualize corpus properties
# generator.plot_statistics('output/plots')
#
# # Step 7: Generate a comprehensive evaluation report
# generator.generate_evaluation_report('output/corpus_evaluation.json')
#
# # Step 8: Print some example sentences from each complexity category
# print("\nExample sentences:")
# for complexity in ['simple', 'medium', 'complex']:
#     print(f"\n{complexity.capitalize()} sentences:")
#     examples = [item for item in corpus if item['metadata']['complexity'] == complexity][:3]
#     for i, example in enumerate(examples, 1):
#         print(f"{i}. {example['sentence']}")
#         print(f"   Avg entropy: {example['metadata']['avg_entropy']:.2f}")



def main():
    parser = argparse.ArgumentParser(description='Generate synthetic corpus for next token prediction tasks')
    parser.add_argument('--output_path', type=str, default='output', help='Path to save the generated corpus')
    parser.add_argument('--output_plot_path', type=str, default='output/plots', help='Path to save the generated corpus plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='next_token_corpus.json', help='Output file name for the corpus')
    parser.add_argument('--corpus_evaluation', type=str, default='corpus_evaluation.json', help='File name for corpus evaluation metrics')
    # parser.add_argument('--include_stats', action='store_true', help='Include statistics in the output corpus')
    parser.add_argument('--corpus_size', type=int, default=25000, help='Number of sentences to generate')
    vocab_sizes = {
        'noun': 2783,  ##
        'transitive_verb': 694,  #
        'intransitive_verb': 694,  #
        'communication_verb': 347,  #
        'motion_verb': 347,  #
        'change_verb': 347,  #
        'adjective': 1388,  ##
        'adverb': 555,  ##
        'location': 694,  ##
        'temporal': 694,  ##
        'preposition': 416,  ##
        'determiner': 111,  ##
        'conjunction': 277,  ##,
        'result': 277,  ##
        # 'comp': 277,##
        # 'rel': 277,##
        # 'result': 277,##
    }
    parser.add_argument('--vocab_sizes', type=json.loads, default=json.dumps(vocab_sizes), help='Vocabulary sizes for different categories')
    parser.add_argument('--complexity_distribution', type=json.loads, default=json.dumps({
        "simple": 0.55,
        "medium": 0.35,
        "complex": 0.10
    }), help='Complexity distribution for sentence generation')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.output_plot_path, exist_ok=True)
    generator = SyntheticCorpusGenerator(args.vocab_sizes)
    corpus = generator.generate_corpus(args.corpus_size, args.complexity_distribution)
    print(f"Generated {len(corpus)} sentences for next token prediction tasks")
    output_file_path = os.path.join(args.output_path, args.output_file)
    generator.save_corpus(output_file_path, include_stats=True)
    evaluation = generator.evaluate_corpus()
    generator.plot_statistics(args.output_plot_path)
    corpus_evaluation_file_path = os.path.join(args.output_path, args.corpus_evaluation)
    generator.generate_evaluation_report(corpus_evaluation_file_path)
    print("Corpus evaluation metrics:")
    for metric, value in evaluation.items():
        print(f"{metric}: {value:.4f}")
    print("\nExample sentences:")
    for complexity in ['simple', 'medium', 'complex']:
        print(f"\n{complexity.capitalize()} sentences:")
        examples = [item for item in corpus if item['metadata']['complexity'] == complexity][:3]
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['sentence']}")
            print(f"   Avg entropy: {example['metadata']['avg_entropy']:.2f}")

if __name__ == "__main__":
    main()
    # todo: improve the labelling of the corpus (SRL, POS, etc.)
    # todo: options to either use templates or supply the templates
    # todo: add option to corrupt the sentences with noise (elimination, substitution, etc.)
