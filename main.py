import json
from collections import Counter

import numpy as np

from synthetic_corpus_generator import SyntheticCorpusGenerator
import os

# Create output directories
os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)

# Step 1: Define vocabulary sizes
vocab_sizes = {
    'noun': 2783,##
     'transitive_verb': 694, #
     'intransitive_verb': 694,#
     'communication_verb': 347,#
     'motion_verb': 347,#
      'change_verb': 347,#
     'adjective': 1388, ##
     'adverb': 555, ##
     'location': 694, ##
     'temporal': 694, ##
     'preposition': 416, ##
     'determiner': 111, ##
     'conjunction': 277, ##,
     'result': 277, ##
     # 'comp': 277,##
     # 'rel': 277,##
    # 'result': 277,##
}
# Step 2: Create generator with custom vocabulary sizes
generator = SyntheticCorpusGenerator(vocab_sizes)

# Step 3: Generate XXX sentences with a specific complexity distribution
# This distribution is optimized for next token prediction tasks
complexity_distribution = {
    "simple": 0.50,     # 55% simple sentences
    "medium": 0.30,     # 35% medium complexity
    "complex": 0.20     # 10% complex with long-distance dependencies
}
corpus_size = 25000  # Number of sentences to generate
corpus = generator.generate_corpus(corpus_size, complexity_distribution)
print(f"Generated {len(corpus)} sentences for original ABSynth corpus")
generator.save_corpus('output/absynth_original.json', include_stats=False)
# corpus = generator.generate_corpus(25000, complexity_distribution)
# print(f"Generated {len(corpus)} sentences")
#
# # Step 4: Save the corpus with statistics
# generator.save_corpus('output/next_token_corpus_4.json', include_stats=True)
# # #
# Step 5: Evaluate corpus for next token prediction suitability
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



# Step 2: Generate the ABSynth-C corpus with 100% complex sentences
print("\nGenerating ABSynth-C corpus (complex sentences only)...")
generator_c = SyntheticCorpusGenerator(vocab_sizes)
complex_only_distribution = {
    "simple": 0.0,
    "medium": 0.0,
    "complex": 1.0  # 100% complex sentences
}
corpus_c = generator_c.generate_corpus(corpus_size, complex_only_distribution)
print(f"Generated {len(corpus_c)} sentences for ABSynth-C corpus")
generator_c.save_corpus('output/absynth_c.json', include_stats=False)

# # Step 5: Evaluate corpus for next token prediction suitability
# evaluation = generator_c.evaluate_corpus()
#
# # Step 6: Generate plots to visualize corpus properties
# generator_c.plot_statistics('output/plots')
#
# # Step 7: Generate a comprehensive evaluation report
# generator_c.generate_evaluation_report('output/corpus_evaluation_c.json')
#
# # Step 8: Print some example sentences from each complexity category
# print("\nExample sentences:")
# for complexity in ['simple', 'medium', 'complex']:
#     print(f"\n{complexity.capitalize()} sentences:")
#     examples = [item for item in corpus_c if item['metadata']['complexity'] == complexity][:3]
#     for i, example in enumerate(examples, 1):
#         print(f"{i}. {example['sentence']}")
#         print(f"   Avg entropy: {example['metadata']['avg_entropy']:.2f}")
#
# Step 3: Generate the ABSynth-LP corpus (low predictability)
# For the low-predictability corpus, we'll generate sentences with a focus on creating
# higher entropy patterns, then filter to keep the highest entropy ones

print("\nGenerating ABSynth-LP corpus (low predictability)...")
generator_lp = SyntheticCorpusGenerator(vocab_sizes)


# Create a custom template manager that favors patterns leading to higher entropy
# This is a technique to generate more sentences with low predictability
class EnhancedTemplateManager:
    def create_low_predictability_templates(self):
        """Create templates that tend to produce higher entropy."""
        return {
            # Templates that create higher entropy by mixing elements in less predictable ways
            "complex": {
                # Mix categories that don't strongly collocate
                ("NOUN", "MOTION_VERB", "PREP", "TEMPORAL", "NOUN", "PREP", "LOCATION"): 0.2,
                # Non-standard word orders
                ("PREP", "LOCATION", "NOUN", "INTRANSITIVE_VERB", "CONJ", "TRANSITIVE_VERB", "NOUN"): 0.2,
                # Complex patterns with varied elements
                ("TEMPORAL", "NOUN", "COMMUNICATION_VERB", "RESULT", "CONJ", "NOUN", "MOTION_VERB", "PREP",
                 "LOCATION"): 0.2,
                # Mix unrelated modifiers
                ("ADJ", "NOUN", "TRANSITIVE_VERB", "RESULT", "PREP", "ADJ", "LOCATION"): 0.2,
                # Complex interleaved patterns
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "LOCATION", "TEMPORAL", "CONJ", "NOUN",
                 "INTRANSITIVE_VERB"): 0.2
            }
        }


# Modify the generator to use custom templates
if hasattr(generator_lp, 'templates'):
    enhanced_templates = EnhancedTemplateManager().create_low_predictability_templates()
    for complexity, templates in enhanced_templates.items():
        for template, weight in templates.items():
            generator_lp.templates.add_custom_template(complexity, template, weight)

# Generate extra sentences to allow for filtering
print("Generating initial set of sentences for ABSynth-LP...")
initial_lp_corpus = generator_lp.generate_corpus(corpus_size * 2, {
    "simple": 0.1,
    "medium": 0.1,
    "complex": 0.8  # Bias toward complex patterns which tend to have higher entropy
})

# Select the highest entropy sentences
print(f"Filtering to select the {corpus_size} highest entropy sentences...")
entropy_values = [(i, item['metadata']['avg_entropy'])
                  for i, item in enumerate(initial_lp_corpus)]
# Sort by entropy in descending order
entropy_values.sort(key=lambda x: x[1], reverse=True)
# Select the top corpus_size indices
top_indices = [index for index, _ in entropy_values[:corpus_size]]
# Create the high-entropy corpus
corpus_lp = [initial_lp_corpus[i] for i in top_indices]

print(f"Created ABSynth-LP corpus with {len(corpus_lp)} sentences")
avg_entropy = np.mean([item['metadata']['avg_entropy'] for item in corpus_lp])
print(f"Average entropy: {avg_entropy:.2f}")


# Save the ABSynth-LP corpus
def save_custom_corpus(corpus_subset, filename):
    """Save a corpus subset with appropriate formatting and statistics."""
    # Calculate basic statistics
    avg_entropy = np.mean([item['metadata']['avg_entropy'] for item in corpus_subset])
    complexity_counts = Counter([item['metadata']['complexity'] for item in corpus_subset])
    total = len(corpus_subset)

    complexity_dist = {
        complexity: count / total for complexity, count in complexity_counts.items()
    }

    # Format for saving
    output = {
        "corpus": corpus_subset,
        "statistics": {
            "corpus_size": len(corpus_subset),
            "avg_entropy": float(avg_entropy),
            "complexity_distribution": complexity_dist
        }
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(corpus_subset)} sentences to '{filename}'")
    print(f"Average entropy: {avg_entropy:.2f}")
    print(f"Complexity distribution: {complexity_dist}")

corpus_lp = corpus_lp['corpus']
for item in corpus_lp:
    del item['metadata']  # Remove metadata for saving

# Save the corpus with statistics
generator_lp.save_corpus('output/absynth_lp.json', include_stats=False)
# Save the ABSynth-LP corpus
# save_custom_corpus(corpus_lp, 'output/absynth_lp.json')