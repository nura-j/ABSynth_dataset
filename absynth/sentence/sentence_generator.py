import random
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

from ..lexicon.semantic_roles import SemanticRole
from ..lexicon.lexicon_generator import LexiconGenerator
from .frame_manager import FrameManager
from .linguistic_annotator import LinguisticAnnotator


class SentenceGenerator:
    """
    Generates sentences with semantic frame structure and rich linguistic annotations.
    """

    def __init__(self, lexicon_generator: Optional[LexiconGenerator] = None, frame_manager: Optional[FrameManager] = None):
        """
        Initialize sentence generator.

        Args:
            lexicon_generator: Instance of LexiconGenerator
            frame_manager: Instance of TemplateManager
        """
        if lexicon_generator:
            self.lexicon = lexicon_generator
        else:
            self.lexicon = LexiconGenerator()
        if frame_manager:
            self.frames = frame_manager
        else:
            self.frames = FrameManager()
        self.annotator = LinguisticAnnotator()

        # Statistics tracking
        self.bigram_counts = Counter()
        self.sentence_lengths = []

    def _expand_frame_argument(self, arg: str, role: SemanticRole, frame_name: str, context: Dict) -> str:
        """
        Expand a frame argument (arg0, arg1, etc.) into an actual word based on semantic role.

        Args:
            arg: Argument identifier (arg0, arg1, etc.)
            role: Semantic role for this argument
            frame_name: Name of the semantic frame
            context: Generation context

        Returns:
            Generated word for this argument
        """
        # Get semantic frame for role-to-POS mapping
        frame = self.lexicon.get_semantic_frame(frame_name)
        if frame and role in frame.pos_mapping:
            pos_category = frame.pos_mapping[role]
        else:
            # Fallback mapping
            pos_mapping = {
                SemanticRole.AGENT: "noun",
                SemanticRole.PATIENT: "noun",
                SemanticRole.THEME: "noun",
                SemanticRole.EXPERIENCER: "noun",
                SemanticRole.INSTRUMENT: "noun",
                SemanticRole.LOCATION: "location",
                SemanticRole.SOURCE: "location",
                SemanticRole.GOAL: "location",
                SemanticRole.TIME: "temporal"
            }
            pos_category = pos_mapping.get(role, "noun")

        return self.lexicon.sample_word(pos_category, context)

    def _expand_template_element(self, element: str, frame_name: str, roles: Dict, context: Dict) -> str:
        """
        Expand any template element (arguments, verbs, prepositions, etc.).

        Args:
            element: Template element to expand
            frame_name: Semantic frame name
            roles: Role mappings for arguments
            context: Generation context

        Returns:
            Expanded word
        """
        # Handle frame arguments (arg0, arg1, etc.)
        if element in roles:
            return self._expand_frame_argument(element, roles[element], frame_name, context)

        # Handle special argument types
        if element == "arg_time":
            return self.lexicon.sample_word("temporal", context)
        elif element == "arg_instr":
            return self.lexicon.sample_word("noun", context)  # Instrument is typically a noun

        # Handle verbs
        if element == "verb":
            verb_mapping = {
                "transitive_action": "transitive_verb",
                "intransitive_action": "intransitive_verb",
                "communication": "communication_verb",
                "motion": "motion_verb"
            }
            verb_type = verb_mapping.get(frame_name, "transitive_verb")
            word = self.lexicon.sample_word(verb_type, context)

            # Apply tense
            if context.get("tense") == "past":
                if random.random() < 0.1:  # 10% irregular verbs
                    return f"{word}{random.choice(['t', 'ed', 'ew', 'ook'])}"
                return f"{word}ed"
            else:
                return f"{word}s" if random.random() > 0.1 else word

        elif element in ["verb1", "verb2"]:
            # For multi-verb templates, use frame-appropriate verbs
            return self._expand_template_element("verb", frame_name, roles, context)

        # Handle other elements
        element_mapping = {
            "adj": "adjective",
            "adv": "adverb",
            "prep": "preposition",
            "conj": "conjunction"
        }

        if element in element_mapping:
            return self.lexicon.sample_word(element_mapping[element], context)

        # Fallback
        return self.lexicon.sample_word(element, context)

    def generate_sentence(self, complexity: Optional[str] = None, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Generate a sentence with semantic frame structure and linguistic annotations.

        Args:
            complexity: Optional complexity level
            include_metadata: Whether to include metadata

        Returns:
            Dictionary with sentence, semantic roles, and linguistic annotations
        """
        # Select template
        if complexity:
            weights = {c: 1.0 if c == complexity else 0.0 for c in ["simple", "medium", "complex"]}
            template_dict, complexity_level = self.frames.select_template(weights)
        else:
            template_dict, complexity_level = self.frames.select_template()

        # Extract template information
        frame_name = template_dict["frame"]
        args_sequence = template_dict["args"]
        roles = template_dict["roles"]

        # Generation context
        context = {"tense": "present", "frame": frame_name}

        # Generate words and track semantic roles
        words = []
        semantic_annotations = {}
        role_assignments = {}

        for i, element in enumerate(args_sequence):
            word = self._expand_template_element(element, frame_name, roles, context)
            words.append(word)

            # Track semantic roles for arguments
            if element in roles:
                role_assignments[element] = {
                    "word": word,
                    "role": roles[element].value,
                    "position": i
                }

            # Update bigram statistics
            if i > 0:
                bigram = (words[i - 1], word)
                self.bigram_counts[bigram] += 1

        # Create sentence
        sentence = " ".join(words)
        self.sentence_lengths.append(len(words))

        # Generate linguistic annotations
        annotations = self.annotator.annotate_sentence(
            sentence=sentence,
            words=words,
            role_assignments=role_assignments,
            frame_name=frame_name
        )

        # Create semantic representation
        semantics = self._create_semantic_representation(role_assignments, frame_name)

        # Calculate entropy profile
        entropy_profile = self._calculate_entropy_profile(words)

        result = {
            "sentence": sentence,
            "semantic_roles": role_assignments,
            "semantics": semantics,
            "linguistic_annotations": annotations.to_dict() if hasattr(annotations, 'to_dict') else annotations
        }

        if include_metadata:
            result["metadata"] = {
                "complexity": complexity_level,
                "frame": frame_name,
                "template": template_dict,
                "length": len(words),
                "entropy_profile": entropy_profile,
                "avg_entropy": sum(entropy_profile) / len(entropy_profile) if entropy_profile else 0
            }

        return result

    def _create_semantic_representation(self, role_assignments: Dict, frame_name: str) -> str:
        """Create formal semantic representation."""
        predicates = []
        for arg, info in role_assignments.items():
            predicates.append(f"{info['role']}(e, {info['word']})")

        if predicates:
            return f"∃e.{frame_name}(e) ∧ {' ∧ '.join(predicates)}"
        return f"∃e.{frame_name}(e)"

    def _calculate_entropy_profile(self, words: List[str]) -> List[float]:
        """
        Calculate entropy profile for next-token prediction based on bigram statistics.
         Inspired by https://stackoverflow.com/questions/990477/how-to-calculate-the-entropy-of-a-file and https://machinelearningmastery.com/what-is-information-entropy/
        """
        entropy_profile = []
        vocab = set(w for bg in self.bigram_counts for w in bg)

        if not vocab:
            vocab = set(words)  # fallback vocab from current sentence
        vocab_size = len(vocab)

        for i in range(len(words) - 1):
            current_word = words[i]
            following_counts = Counter({w: 1 for w in vocab})  # we start with 1 count for each word to avoid zero probabilities
            total_count = vocab_size

            for (w1, w2), count in self.bigram_counts.items():
                if w1 == current_word:
                    following_counts[w2] += count
                    total_count += count

            probs = [count / total_count for count in following_counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            entropy_profile.append(entropy)

        # Fallback: use mean of previous entropies or default to log2(V) if none
        if entropy_profile:
            final_entropy = sum(entropy_profile) / len(entropy_profile)
        else:
            final_entropy = math.log2(vocab_size) if vocab_size > 0 else 1.0

        entropy_profile.append(final_entropy)  # For sentence-end token

        return entropy_profile


    def get_predictability_distribution(
            self,
            entropy_profiles: List[List[float]],
            thresholds: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate predictability distribution across corpus.

        Args:
            entropy_profiles: List of entropy profiles for sentences
            thresholds: Tuple of (high_threshold, low_threshold) for predictability bins.
                        Default is adaptive based on quantiles.

        Returns:
            Dictionary with proportions of high, medium, and low predictability tokens.
        """
        entropy_values = [e for profile in entropy_profiles for e in profile]
        if not entropy_values:
            return {
                "high_predictability": 0.0,
                "medium_predictability": 0.0,
                "low_predictability": 0.0
            }

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
            "high_predictability": high_pred,
            "medium_predictability": medium_pred,
            "low_predictability": low_pred
        }

