# ABSynth: Advanced Synthetic Corpus Generator

**ABSynth** is a Python library for generating high-quality synthetic corpora with controlled linguistic properties, semantic frame annotations, and statistical characteristics optimized for NLP tasks, particularly next token prediction.

## Features

- **Semantic Frame Integration**: Generate sentences with rich semantic role annotations (Agent, Patient, Theme, Location, etc.)
- **Statistical Control**: Precise control over complexity distributions, Zipfian word frequencies, and entropy profiles
- **NLP Optimization**: Specifically designed for next token prediction and language model training, and easly extendable for other NLP tasks
- **Rich Annotations**: POS tags, dependency parsing, constituency trees, and formal semantics
- **Multiple Formats**: Export to JSON and specialized formats for different use cases
- **Scalable Generation**: Efficient generation of large corpora with consistent quality
- **Visualization Tools**: Analyze corpus properties, distributions, and semantic diversity
- **Intervention Framework**: Perform automated perturbations like synonymic substitution, semantic role violation, and argument elimination for probing model robustness


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nura-j/absynth.git
cd absynth

# [Optional] create and activate the conda environment
conda env create -f environment.yml
conda activate ABSYN_dataset

# Install
pip install .
```

### Basic Usage

```python
from absynth.corpus import SyntheticCorpusGenerator

# Create generator with default settings
generator = SyntheticCorpusGenerator()

# Generate a corpus with 10K sentences with default distributions
corpus = generator(10000)

# Checking distributions
print(corpus.complexity_distribution)  # {'simple': 0.55, 'medium': 0.35, 'complex': 0.1}
print(corpus.semantic_frame_distribution)  # {'basic_transitive': 0.25, 'basic_intransitive': 0.2, ... }

# Save corpus
corpus.save("corpus_full.json", indent=2)

# Export in different formats
corpus.export("corpus_semantic.json", format="semantic_annotations", indent=2) 
corpus.export("corpus_sentences.json", format="sentences_only", indent=2)

# Evaluate quality
evaluation = generator.evaluate_corpus(corpus, calculate_suitability=True)
print(f"Suitability Score: {evaluation['suitability']['suitability_score']:.2f}")

# Visualize the corpus statistics
from absynth.visualization import Visualizer
visualizer = Visualizer(log_dir='./plots')
visualizer.visualize(corpus)

```

```python
from absynth.corpus import SyntheticCorpusGenerator

# Create generator with default settings
generator = SyntheticCorpusGenerator()

# Generate a corpus with custom distributions
corpus = generator.generate_corpus(
    num_sentences=10000,
    complexity_distribution={"simple": 0.55, "medium": 0.35, "complex": 0.1},
    semantic_frame_distribution={
        "transitive_action": 0.4,
        "intransitive_action": 0.25,
        "communication": 0.2,
        "motion": 0.15
    }
)
```

```python
from absynth.corpus import SynthCorpus

# Load saved corpus
corpus = SynthCorpus.load("corpus_full.json")
```

## Documentation

### Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Lexicon System](#lexicon-system)
3. [Sentence Generation](#sentence-generation)
4. [Corpus Generation](#corpus-generation)
5. [Analysis Tools](#analysis-tools)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Intervention Framework](#intervention-framework)
8. [Contributing](#contributing)
9. [License](#license)
10. [Citation](#citation)

## Architecture Overview

ABSynth consists of four main packages:

```
absynth/
├── lexicon/           # Vocabulary management and word generation
├── sentence/          # Template-based sentence generation with semantic frames
├── corpus/            # Large-scale corpus generation and evaluation
├── intervention/       # Automated perturbation framework for generated corpus
└── visualization/     # Visualization tools for corpus analysis    
```

### Core Components

- **LexiconGenerator**: Manages vocabulary with Zipfian distributions
- **FrameManager**: Handles semantic frame templates and role assignments
- **SentenceGenerator**: Creates annotated sentences with linguistic features
- **SyntheticCorpusGenerator**: Orchestrates large-scale corpus generation
- **SynthCorpus**: Provides core corpus data, metadata and functionalities
- **Visualizer**: Visualizes corpus statistics and distributions
- **InterventionManager**: Applies perturbations to corpus for robustness testing
- **CorpusEvaluator**: Provides quality assessment and NLP suitability metrics

## Lexicon System

### Basic Vocabulary Configuration

```python
from absynth.lexicon import Vocabulary, LexiconGenerator

# Define vocabulary sizes
vocab = Vocabulary({
    "noun":300,
    "transitive_verb":40,
    "intransitive_verb":25,
    "communication_verb":20,
    "motion_verb":20,
    "change_verb":15,
    "adjective":40,
    "adverb":25,
    "location":150,
    "temporal":35,
    "instrument":25,
    "preposition":15,
    "conjunction":10,
    "determiner":8
})

# Create lexicon generator with all features
lexicon = LexiconGenerator(
    vocab_sizes=vocab,           # Custom vocabulary sizes
    num_clusters=5,              # Number of semantic clusters to create
    zipfian_alpha=1.05,             # Alpha parameter for Zipfian distribution
    error_bias=0.00001,              # Error bias for word generation
    random_seed=42               # For reproducible generation
)

# Alternatively, use defaults for simpler initialization
# lexicon = LexiconGenerator()  # Uses built-in vocabulary and default parameters
# Check what vocabulary was created
print(lexicon.vocab_sizes)
```
### Semantic Clusters and Collocations
ABSynth automatically organizes words into semantic clusters and establishes collocational relationships:
```python
# Access semantic clusters
clusters = lexicon.semantic_clusters
print(f"Number of noun clusters: {len(clusters['noun'])}")
print(f"Words in first noun cluster: {clusters['noun']['cluster_0']}")

# Access collocations between words
collocations = lexicon.collocations
# Get collocation strength between a noun and adjective
noun = lexicon.sample_word("noun")
adj = lexicon.sample_word("adjective")
strength = collocations[noun].get(adj, 0)
print(f"Collocation strength between '{noun}' and '{adj}': {strength:.2f}")
# Ranges: 0.4-0.7 (strong), 0.3-0.6 (medium), 0.05-0.25 (weak)
```

### Advanced Lexicon Features
```python
# Sample words with context awareness
context = {"force_low_predictability": True}
unpredictable_word = lexicon.sample_word("noun", context=context)

# Export lexicon for analysis
lexicon_data = lexicon.export_lexicon_details()
print(f"Total vocabulary size: {sum(len(words) for words in lexicon.lexicon.values())}")

# Add custom semantic frames
from absynth.semantics import SemanticFrame, SemanticRole
custom_frame = SemanticFrame(
    frame_name="perception",
    core_roles=[SemanticRole.EXPERIENCER, SemanticRole.STIMULUS],
    optional_roles=[SemanticRole.INSTRUMENT, SemanticRole.MANNER],
    pos_mapping={
        SemanticRole.EXPERIENCER: "noun",
        SemanticRole.STIMULUS: "noun",
        SemanticRole.INSTRUMENT: "noun",
        SemanticRole.MANNER: "adverb"
    }
)
lexicon.add_semantic_frame(custom_frame)
```

##  Sentence Generation

### Semantic Frames

ABSynth supports multiple semantic frames with rich role annotations:

- **Transitive Action**: Agent performs action on Patient
- **Intransitive Action**: Agent performs action
- **Communication**: Agent communicates Message to Recipient
- **Motion**: Theme moves from Source to Goal
- **Change**: Agent causes Theme to change State
- **Instrumental Action**: Agent uses Instrument to affect Patient
- **Transfer**: Agent gives Theme to Goal
- **Coordination**: Multiple Agents or Patients in coordinated actions

### Example Generation

```python
from absynth.sentence import SentenceGenerator, FrameManager
from absynth.lexicon import LexiconGenerator

# Initialize components
lexicon = LexiconGenerator()
templates = FrameManager()
generator = SentenceGenerator(lexicon, templates)

# or create a SentenceGenerator with default settings
# generator = SentenceGenerator()

# Generate sentence with specific complexity
sentence_data = generator.generate_sentence(
    complexity="medium",
    include_metadata=True
)

# or generate a sentence with random complexity
# sentence_data = generator.generate_sentence(
#     include_metadata=True
# )

print(sentence_data)
# Output:
# {'sentence': 'noun139 transitive_verb8s noun40 preposition4 location2', 
#  'semantic_roles': {'arg0': {'word': 'noun139', 'role': 'Agent', 'position': 0}, 
#                     'arg1': {'word': 'noun40', 'role': 'Patient', 'position': 2}, 
#                     'arg2': {'word': 'location2', 'role': 'Location', 'position': 4}}, 
#  'semantics': '∃e.transitive_action(e) ∧ Agent(e, noun139) ∧ Patient(e, noun40) ∧ Location(e, location2)', 
#  'linguistic_annotations': {'pos_tags': ['NN', 'VB', 'NN', 'IN', 'NN'], 
#                             'semantic_roles': {'noun139': 'Agent', 'noun40': 'Patient', 'location2': 'Location'}, 
#                             'formal_semantics': 'λx0 x2 x4.∃e.transitive_action(e) ∧ Agent(x0) ∧ Patient(x2) ∧ Location(x4)'}, 
#  'metadata': {'complexity': 'medium', 
#               'frame': 'transitive_action', 'template': {'frame': 'transitive_action', 'args': ['arg0', 'verb', 'arg1', 'prep', 'arg2'], 'roles': {'arg0': <SemanticRole.AGENT: 'Agent'>, 'arg1': <SemanticRole.PATIENT: 'Patient'>, 'arg2': <SemanticRole.LOCATION: 'Location'>}, 'weight': 0.2}, 
# 'length': 5, 'entropy_profile': [2.2516291673878226, 2.2516291673878226, 2.2516291673878226, 2.2516291673878226, 2.2516291673878226], 
# 'avg_entropy': 2.2516291673878226}}


```

## Corpus Generation

### Large-Scale Generation

```python
from absynth.corpus import SyntheticCorpusGenerator

generator = SyntheticCorpusGenerator()

# Generate large corpus with specific distributions
corpus = generator.generate_corpus(
    num_sentences=50000,
    complexity_distribution={
        "simple": 0.45,
        "medium": 0.35, 
        "complex": 0.2
    },
    semantic_frame_distribution={
        "transitive_action": 0.4,
        "intransitive_action": 0.25,
        "communication": 0.2,
        "motion": 0.15
    },
    include_annotations=True
)
# or generate a corpus with random complexity and frame distribution
# Generate corpus with default settings
# corpus = generator.generate_corpus(num_sentences=10000)

print(f"Generated {len(corpus):,} sentences")
```

### Output Formats

```python
# Full format with all annotations
# Save corpus
corpus.save("corpus_full.json", indent=2)

# Export in different formats
corpus.export("corpus_semantic.json", format="semantic_annotations", indent=2) 
corpus.export("corpus_sentences.json", format="sentences_only", indent=2)

```


## Analysis Tools

### Quick Quality Assessment

```python
# Quick evaluation
evaluation = generator.evaluate_corpus(calculate_suitability=True)

print(f"Suitability Score: {evaluation['suitability']['suitability_score']:.2f}")
print(f"Category: {evaluation['suitability']['suitability_category']}")
print(f"Semantic Diversity: {evaluation['semantic_analysis']['frame_diversity']:.3f}")

# Recommendations for improvement
for rec in evaluation['suitability']['recommendations']:
    print(f"• {rec}")
```
## Evaluation Metrics

ABSynth provides comprehensive evaluation metrics:

### Statistical Metrics
- **Type-Token Ratio**: Vocabulary diversity measure
- **Zipfian Compliance**: How well word frequencies follow Zipf's law
- **Entropy Distribution**: Predictability patterns for next token prediction
- **Sentence Length Statistics**: Mean, standard deviation, distribution

### Semantic Metrics
- **Frame Diversity**: Number of unique semantic frames used
- **Role Coverage**: Percentage of semantic roles covered

### NLP Suitability Metrics
- **Overall Suitability Score**: 0-1 score for next token prediction readiness
- **Predictability Balance**: Distribution of high/medium/low predictability contexts
- **Recommendations**: Specific suggestions for improvement


## Intervention 

ABSynth supports applying controlled interventions for interpretability, robustness, and probing studies:

### Supported Interventions
- **Synonymic Substitution:** Replace an argument with a semantically similar token while preserving the same role.
- **Role Violation:** Replace an argument with a semantically inconsistent token to create a role mismatch.
- **Argument Elimination:** Remove an argument and adjacent function words to test role sensitivity.

```python
from absynth.intervention import (
    apply_synonymic_substitution,
    apply_role_violation,
    apply_elimination,
    InterventionManager
)

# Apply interventions
synonymic_results = apply_synonymic_substitution(corpus, subset_percentage=0.3, save_results=True, save_path="synonymic_substitution_results.json")
violation_results = apply_role_violation(corpus, subset_percentage=0.2) # if save_results=False, results are not saved
elimination_results = apply_elimination(corpus, subset_percentage=0.2)

# Export datasets
intervention = InterventionManager(corpus, "role_violation")
intervention.export_intervention_dataset(violation_results, "role_violation_dataset.json")
```

### Output Schema
The exported JSON includes:
- Original and modified sentence
- Type of intervention
- Target role and token replaced
- Position index
- Rich metadata (e.g. semantic frame, success flag)

```json
{
  "intervention_type": "role_violation",
  "intervention_statistics": {
    "total": 500,
    "successful": 472,
    "failure_rate": 0.056
  },
  "results": [
    {
      "original_sentence": "noun3 transitive_verb12 noun14",
      "modified_sentence": "noun3 transitive_verb12 location5",
      "intervention_type": "role_violation",
      "target_role": "Patient",
      "target_word": "noun14",
      "replacement_word": "location5",
      "position": 2,
      "metadata": {
        "sentence_idx": 3,
        "semantic_frame": "transitive_action",
        "violation_type": "role_mismatch",
        "intervention_success": true
      }
    }
  ]
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.


##  License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ABSynth in your research, please cite:


```bibtex
@misc{aljaafari2025tracetraininginferencetimeinterpretability,
      title={TRACE: Training and Inference-Time Interpretability Analysis for Language Models}, 
      author={Nura Aljaafari and Danilo S. Carvalho and André Freitas},
      year={2025},
      eprint={2507.03668},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.03668}, 
}
```