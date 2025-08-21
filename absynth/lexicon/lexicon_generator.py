import numpy as np
from collections import defaultdict, Counter
import random
from typing import Dict, List, Any, Optional
from .vocab import Vocabulary
from .semantic_roles import SemanticFrame, SemanticRoles


class LexiconGenerator:
    """
    Generates a synthetic lexicon with realistic distributional properties
    for next token prediction tasks.
    """
    
    def __init__(self, vocab_sizes: Vocabulary = None,
                 num_clusters: int = 5,
                 additional_semantic_frames: Optional[List[SemanticFrame]] = None,
                 zipfian_alpha: float = 1.05,
                 error_bias: float = 0.0000001,
                 random_seed: Optional[int] = None):
        """
        Initialize with vocabulary sizes for each word category.
        
        Args:
            vocab_sizes: Dictionary mapping word categories to vocabulary sizes
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.vocab_sizes = vocab_sizes or Vocabulary()
        self._num_clusters = num_clusters
        self.zipfian_alpha = zipfian_alpha
        self.error_bias = error_bias

        self.lexicon = self._generate_lexicon()
        self.semantic_clusters = self._create_semantic_clusters()
        self.word_probabilities = self._create_zipfian_distributions()
        self.collocations = self._establish_collocations()

        self.semantic_frames = SemanticRoles.get_standard_frames()
        if additional_semantic_frames:
            for frame in additional_semantic_frames:
                self.add_semantic_frame(frame)
        # Usage tracking
        self.used_words = {cat: set() for cat in self.word_probabilities.keys()} #todo: still need to track used words when sampling
        self.word_counts = {cat: Counter() for cat in self.word_probabilities.keys()} #todo: still need to track word counts when sampling

        # Initialize entropy tracking
        self.entropy_counts = {
            'high_predictability': 0,  # entropy < 1.5
            'medium_predictability': 0,  # 1.5 <= entropy < 3.0
            'low_predictability': 0  # entropy >= 3.0
        }
        self.total_entropy_samples = 0

    def _generate_lexicon(self) -> Dict[str, List[str]]:
        """Create a lexicon with synthetic words for each category.
        We use POS tags as prefixes to create unique words, though this can be changed to any words selection"""

        lexicon = {}
        for pos, size in self.vocab_sizes.items():
            lexicon[pos] = [f"{pos}{i}" for i in range(1, size + 1)]
        return lexicon

    def _adjust_for_target_entropy_distribution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust the context to ensure proper entropy distribution in the corpus.

        This forces some percentage of transitions to have low predictability (high entropy).

        Args:
            context: The current generation context

        Returns:
            Updated context
        """

        # Calculate current distribution
        total = sum(self.entropy_counts.values())
        if total > 100:  # Only start adjusting after gathering some statistics
            low_pred_ratio = self.entropy_counts['low_predictability'] / total

            # If we're below target for low predictability (target is 20%)
            if low_pred_ratio < 0.2:
                # Increase chance of forcing low predictability
                context['force_low_predictability'] = True
            else:
                context['force_low_predictability'] = False

        return context

    def _create_zipfian_distributions(self) -> Dict[str, Dict[str, float]]:
        """Create Zipfian probability distributions for words in each category."""
        distributions = {}
        
        for category, words in self.lexicon.items():
            # Generate ranks
            ranks = np.arange(1, len(words) + 1)
            # Calculate probabilities using Zipf's law
            probs = 1.0 / (ranks ** self.zipfian_alpha)
            # Normalize to sum to 1
            probs = (probs+self.error_bias) / (probs.sum()+self.error_bias)
            
            # Map words to probabilities
            distributions[category] = {word: prob for word, prob in zip(words, probs)}
        
        return distributions
    
    def _create_semantic_clusters(self) -> Dict[str, Dict[str, List[str]]]:
        """Create semantic clusters for realistic word associations."""
        clusters = {}
        
        # Create noun clusters
        if "noun" in self.lexicon:
            nouns = self.lexicon["noun"]
            noun_clusters = np.array_split(nouns, self._num_clusters)  # Split nouns into clusters
            clusters["noun"] = {
                f"cluster_{i}": list(cluster) for i, cluster in enumerate(noun_clusters)
            }
        
        # Create adjective clusters that semantically align with noun clusters
        if "adjective" in self.lexicon:
            adjectives = self.lexicon["adjective"]
            adj_clusters = np.array_split(adjectives, self._num_clusters) # Split nouns into clusters
            clusters["adjective"] = {
                f"cluster_{i}": list(cluster) for i, cluster in enumerate(adj_clusters)
            }
        
        # Create verb clusters
        for verb_type in ["transitive_verb", "intransitive_verb", "communication_verb", "motion_verb"]:
            if verb_type in self.lexicon and len(self.lexicon[verb_type]) >= self._num_clusters:
                verbs = self.lexicon[verb_type]
                verb_clusters = np.array_split(verbs, self._num_clusters)
                clusters[verb_type] = {
                    f"cluster_{i}": list(cluster) for i, cluster in enumerate(verb_clusters)
                }
        
        return clusters


    def _establish_collocations(self) -> Dict[str, Dict[str, float]]:
        """Establish collocational preferences between words for realistic co-occurrence.
        To ensure a balanced distribution of predictability, we create collocations
        with varying strengths based on semantic clusters and word types.
        This method creates collocations between nouns, adjectives, and verbs
        with a focus on realistic predictability distributions.
        The collocations are designed to reflect natural language patterns,
        where some word pairs are strongly associated, while others are weakly associated.
        The collocations are created with the following principles:
        1. Strong associations between nouns and adjectives within the same semantic cluster.
        2. Medium-strength associations between verbs and nouns, with some variation.
        3. Weaker associations between words in different clusters to reflect natural language diversity.
        4. Adverbs are associated with verbs, but with medium strength to avoid predictability.
        5. The collocations are designed to reflect a range of predictability, from strong to very weak,
        ensuring a realistic distribution of word associations.
        """
        collocations = defaultdict(dict)
        # Define collocation strength constants
        STRONG_BASE, STRONG_VARIANCE = 0.4, 0.3  # Range 0.4-0.7
        MEDIUM_BASE, MEDIUM_VARIANCE = 0.3, 0.3  # Range 0.3-0.6
        WEAK_BASE, WEAK_VARIANCE = 0.05, 0.2  # Range 0.05-0.25
        VERY_WEAK_BASE, VERY_WEAK_VARIANCE = 0.05, 0.15  # Range 0.05-0.2

        # Create adjective-noun (NP) collocations with more variety
        if "noun" in self.semantic_clusters and "adjective" in self.semantic_clusters:
            for i in range(len(self.semantic_clusters["noun"])):
                noun_cluster = self.semantic_clusters["noun"][f"cluster_{i}"]
                adj_cluster = self.semantic_clusters["adjective"][f"cluster_{i}"]

                # Each noun has stronger association with adjectives in the same cluster
                for noun in noun_cluster:
                    # Strong association with adjectives in the same cluster, but more varied
                    for adj in adj_cluster:
                        # Add noise to create more medium-predictability contexts
                        collocations[noun][adj] = STRONG_BASE + random.random() * STRONG_VARIANCE  # Range 0.4-0.7

                    # Weaker association with adjectives in other clusters, but still exist
                    for j in range(len(self.semantic_clusters["adjective"])):
                        if j != i:
                            for adj in self.semantic_clusters["adjective"][f"cluster_{j}"]:
                                # Add noise for more natural distributions
                                collocations[noun][adj] = VERY_WEAK_BASE + random.random() * VERY_WEAK_VARIANCE  # Range 0.05-0.2

        # Create verb-noun (object) collocations with more balanced predictability
        if "noun" in self.semantic_clusters:
            for verb_type in ["transitive_verb", "intransitive_verb"]:
                if verb_type in self.lexicon:
                    verbs = self.lexicon[verb_type]
                    verb_clusters = np.array_split(verbs, len(self.semantic_clusters["noun"])) # Split verbs into clusters

                    for i, verb_c in enumerate(verb_clusters):
                        noun_cluster = self.semantic_clusters["noun"][f"cluster_{i}"]
                        for verb in verb_c:
                            for noun in noun_cluster:
                                collocations[verb][noun] = MEDIUM_BASE + random.random() * MEDIUM_VARIANCE # Range 0.3-0.6

                            # Connections to other clusters but weaker
                            for j in range(len(self.semantic_clusters["noun"])):
                                if j != i:
                                    for noun in self.semantic_clusters["noun"][f"cluster_{j}"]:
                                        collocations[verb][noun] = WEAK_BASE + random.random() * WEAK_VARIANCE # Range 0.05-0.25

        # Add collocations for adverbs and verbs (VP)
        if "adverb" in self.lexicon and "transitive_verb" in self.lexicon:
            adverbs = self.lexicon["adverb"]
            verbs = self.lexicon["transitive_verb"] + self.lexicon.get("intransitive_verb", [])

            # Create adverb clusters aligned with verb semantics
            adverb_clusters = np.array_split(adverbs, min(self._num_clusters, len(adverbs)))
            verb_clusters = np.array_split(verbs, min(self._num_clusters, len(adverbs)))

            for i in range(min(len(adverb_clusters), len(verb_clusters))):
                adv_cluster = adverb_clusters[i]
                verb_cluster = verb_clusters[i]

                for verb in verb_cluster:
                    for adv in adv_cluster:
                        # Medium-strength collocations (not too predictable, not too random)
                        collocations[verb][adv] = MEDIUM_BASE + random.random() * WEAK_VARIANCE

        return collocations

    def sample_word(self, category, context=None) -> str:
        """
        Sample a word with entropy-aware selection to ensure balanced predictability.

        Args:
            category: Word category to sample from
            context: Optional context information including entropy targets

        Returns:
            A sampled word
        """
        # Default context
        if context is None:
            context = {}

        # Check if we should force low predictability
        force_low_pred = context.get('force_low_predictability', False)

        # Handle categories not in lexicon
        if category not in self.word_probabilities:
            if force_low_pred:
                # Use a larger range for more unpredictability
                return f"{category}{random.randint(1, 1000)}"
            else:
                return f"{category}{random.randint(1, 10)}"

        # Get available words for the category
        available_words = list(self.word_probabilities[category].keys())

        # Force low predictability by selecting random words
        if force_low_pred and random.random() < 0.8:  # 80% chance when forcing
            # Completely random selection to maximize entropy
            return random.choice(available_words)

        # Track used words if not already tracking
        if not hasattr(self, 'used_words'):
            self.used_words = {cat: set() for cat in self.word_probabilities.keys()}
            self.word_counts = {cat: Counter() for cat in self.word_probabilities.keys()}

        # Ensure category is being tracked
        if category not in self.used_words:
            self.used_words[category] = set()
            self.word_counts[category] = Counter()

        # Prioritize unused words (70% chance of using a new word if available)
        unused_words = [w for w in available_words if w not in self.used_words[category]]
        if unused_words and random.random() < 0.7:
            word = random.choice(unused_words)
        else:
            # Use inverse frequency weighting to prioritize less common words
            weights = []
            for word in available_words:
                count = self.word_counts[category].get(word, 0)
                # Higher weight for less used words
                weight = 1.0 / (count + 1)
                weights.append(weight)

            # Sample using inverse frequency weights
            word = random.choices(available_words, weights=weights)[0]

        # Update usage statistics
        self.used_words[category].add(word)
        self.word_counts[category][word] += 1

        return word

    def get_semantic_frame(self, frame_name: str) -> Optional[SemanticFrame]:
        """Get a semantic frame by name."""
        return self.semantic_frames.get(frame_name, None)

    def add_semantic_frame(self, frame: SemanticFrame) -> None:
        """Add a new semantic frame to the lexicon."""
        if frame.frame_name in self.semantic_frames:
            raise ValueError(f"Frame '{frame.frame_name}' already exists.")
        self.semantic_frames[frame.frame_name] = frame

    def export_lexicon_details(self) -> Dict:
        """Export comprehensive lexicon information."""
        return {
            "vocabulary_sizes": dict(self.vocab_sizes.items()),
            "lexicon": self.lexicon,
            "zipfian_distributions": {
                category: {word: float(prob) for word, prob in probs.items()}
                for category, probs in self.word_probabilities.items()
            },
            "semantic_clusters": self.semantic_clusters,
            "semantic_frames": {
                name: {
                    "frame_name": frame.frame_name,
                    "core_roles": [role.value for role in frame.core_roles],
                    "optional_roles": [role.value for role in frame.optional_roles],
                    "pos_mapping": {role.value: pos for role, pos in frame.pos_mapping.items()}
                }
                for name, frame in self.semantic_frames.items()
            }
        }

    # todo: load lexicon from file