import random
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

from ..lexicon.semantic_roles import SemanticRole
from .linguistic_annotator import LinguisticAnnotator


class SentenceGenerator:
    """
    Generates sentences with semantic frame structure and rich linguistic annotations.
    """

    def __init__(self, lexicon_generator, template_manager):
        """
        Initialize sentence generator.

        Args:
            lexicon_generator: Instance of LexiconGenerator
            template_manager: Instance of TemplateManager
        """
        self.lexicon = lexicon_generator
        self.templates = template_manager
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
            template_dict, complexity_level = self.templates.select_template(weights)
        else:
            template_dict, complexity_level = self.templates.select_template()

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
        """Calculate entropy profile for next token prediction."""
        entropy_profile = []

        for i in range(len(words) - 1):
            current_word = words[i]
            following_counts = Counter()
            total_count = 0

            for bg, count in self.bigram_counts.items():
                if bg[0] == current_word:
                    following_counts[bg[1]] += count
                    total_count += count

            if total_count > 0:
                probs = [count / total_count for count in following_counts.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)

                # Adjust toward medium range
                if entropy < 1.0:
                    entropy = 1.0 + (entropy * 0.5)
                elif entropy > 3.0:
                    entropy = 3.0 - ((entropy - 3.0) * 0.5)

                entropy_profile.append(entropy)
            else:
                entropy_profile.append(2.0)  # Medium predictability

        entropy_profile.append(2.8)  # End of sentence
        return entropy_profile

    def get_predictability_distribution(self, entropy_profiles: List[List[float]]) -> Dict[str, float]:
        """Calculate predictability distribution across corpus."""
        entropy_values = []
        for profile in entropy_profiles:
            entropy_values.extend(profile)

        if not entropy_values:
            return {"high_predictability": 0, "medium_predictability": 0, "low_predictability": 0}

        total = len(entropy_values)
        high_pred = sum(1 for e in entropy_values if e < 1.5) / total
        medium_pred = sum(1 for e in entropy_values if 1.5 <= e < 3.0) / total
        low_pred = sum(1 for e in entropy_values if e >= 3.0) / total

        return {
            "high_predictability": high_pred,
            "medium_predictability": medium_pred,
            "low_predictability": low_pred
        }
# class SentenceGenerator:
#     """
#     Generates sentences with semantic frame structure and rich linguistic annotations.
#     """
#
#     def __init__(self, lexicon_generator, template_manager):
#         """
#         Initialize sentence generator.
#
#         Args:
#             lexicon_generator: Instance of LexiconGenerator
#             template_manager: Instance of TemplateManager
#         """
#         self.lexicon = lexicon_generator
#         self.templates = template_manager
#         self.annotator = LinguisticAnnotator()
#
#         # Statistics tracking
#         self.bigram_counts = Counter()
#         self.sentence_lengths = []
#
#     def _expand_frame_argument(self, arg: str, role: SemanticRole, frame_name: str, context: Dict) -> str:
#         """
#         Expand a frame argument (arg0, arg1, etc.) into an actual word based on semantic role.
#
#         Args:
#             arg: Argument identifier (arg0, arg1, etc.)
#             role: Semantic role for this argument
#             frame_name: Name of the semantic frame
#             context: Generation context
#
#         Returns:
#             Generated word for this argument
#         """
#         # Get semantic frame for role-to-POS mapping
#         frame = self.lexicon.get_semantic_frame(frame_name)
#         if frame and role in frame.pos_mapping:
#             pos_category = frame.pos_mapping[role]
#         else:
#             # Fallback mapping
#             pos_mapping = {
#                 SemanticRole.AGENT: "noun",
#                 SemanticRole.PATIENT: "noun",
#                 SemanticRole.THEME: "noun",
#                 SemanticRole.EXPERIENCER: "noun",
#                 SemanticRole.INSTRUMENT: "noun",
#                 SemanticRole.LOCATION: "location",
#                 SemanticRole.SOURCE: "location",
#                 SemanticRole.GOAL: "location",
#                 SemanticRole.TIME: "temporal"
#             }
#             pos_category = pos_mapping.get(role, "noun")
#
#         return self.lexicon.sample_word(pos_category, context)
#
#     def _expand_template_element(self, element: str, frame_name: str, roles: Dict, context: Dict) -> str:
#         """
#         Expand any template element (arguments, verbs, prepositions, etc.).
#
#         Args:
#             element: Template element to expand
#             frame_name: Semantic frame name
#             roles: Role mappings for arguments
#             context: Generation context
#
#         Returns:
#             Expanded word
#         """
#         # Handle frame arguments (arg0, arg1, etc.)
#         if element in roles:
#             return self._expand_frame_argument(element, roles[element], frame_name, context)
#
#         # Handle special argument types
#         if element == "arg_time":
#             return self.lexicon.sample_word("temporal", context)
#         elif element == "arg_instr":
#             return self.lexicon.sample_word("noun", context)  # Instrument is typically a noun
#
#         # Handle verbs
#         if element == "verb":
#             verb_mapping = {
#                 "transitive_action": "transitive_verb",
#                 "intransitive_action": "intransitive_verb",
#                 "communication": "communication_verb",
#                 "motion": "motion_verb"
#             }
#             verb_type = verb_mapping.get(frame_name, "transitive_verb")
#             word = self.lexicon.sample_word(verb_type, context)
#
#             # Apply tense
#             if context.get("tense") == "past":
#                 if random.random() < 0.1:  # 10% irregular verbs
#                     return f"{word}{random.choice(['t', 'ed', 'ew', 'ook'])}"
#                 return f"{word}ed"
#             else:
#                 return f"{word}s" if random.random() > 0.1 else word
#
#         elif element in ["verb1", "verb2"]:
#             # For multi-verb templates, use frame-appropriate verbs
#             return self._expand_template_element("verb", frame_name, roles, context)
#
#         # Handle other elements
#         element_mapping = {
#             "adj": "adjective",
#             "adv": "adverb",
#             "prep": "preposition",
#             "conj": "conjunction"
#         }
#
#         if element in element_mapping:
#             return self.lexicon.sample_word(element_mapping[element], context)
#
#         # Fallback
#         return self.lexicon.sample_word(element, context)
#
#     def generate_sentence(self, complexity: Optional[str] = None, include_metadata: bool = True) -> Dict[str, Any]:
#         """
#         Generate a sentence with semantic frame structure and linguistic annotations.
#
#         Args:
#             complexity: Optional complexity level
#             include_metadata: Whether to include metadata
#
#         Returns:
#             Dictionary with sentence, semantic roles, and linguistic annotations
#         """
#         # Select template
#         if complexity:
#             weights = {c: 1.0 if c == complexity else 0.0 for c in ["simple", "medium", "complex"]}
#             template_dict, complexity_level = self.templates.select_template(weights)
#         else:
#             template_dict, complexity_level = self.templates.select_template()
#
#         # Extract template information
#         frame_name = template_dict["frame"]
#         args_sequence = template_dict["args"]
#         roles = template_dict["roles"]
#
#         # Generation context
#         context = {"tense": "present", "frame": frame_name}
#
#         # Generate words and track semantic roles
#         words = []
#         semantic_annotations = {}
#         role_assignments = {}
#
#         for i, element in enumerate(args_sequence):
#             word = self._expand_template_element(element, frame_name, roles, context)
#             words.append(word)
#
#             # Track semantic roles for arguments
#             if element in roles:
#                 role_assignments[element] = {
#                     "word": word,
#                     "role": roles[element].value,
#                     "position": i
#                 }
#
#             # Update bigram statistics
#             if i > 0:
#                 bigram = (words[i - 1], word)
#                 self.bigram_counts[bigram] += 1
#
#         # Create sentence
#         sentence = " ".join(words)
#         self.sentence_lengths.append(len(words))
#
#         # Generate linguistic annotations
#         annotations = self.annotator.annotate_sentence(
#             sentence=sentence,
#             words=words,
#             role_assignments=role_assignments,
#             frame_name=frame_name
#         )
#
#         # Create semantic representation
#         semantics = self._create_semantic_representation(role_assignments, frame_name)
#
#         # Calculate entropy profile
#         entropy_profile = self._calculate_entropy_profile(words)
#
#         result = {
#             "sentence": sentence,
#             "semantic_roles": role_assignments,
#             "semantics": semantics,
#             "linguistic_annotations": annotations.to_dict()  # Convert to dict for JSON serialization
#         }
#
#         if include_metadata:
#             result["metadata"] = {
#                 "complexity": complexity_level,
#                 "frame": frame_name,
#                 "template": template_dict,
#                 "length": len(words),
#                 "entropy_profile": entropy_profile,
#                 "avg_entropy": sum(entropy_profile) / len(entropy_profile) if entropy_profile else 0
#             }
#
#         return result
#
#     def _create_semantic_representation(self, role_assignments: Dict, frame_name: str) -> str:
#         """Create formal semantic representation."""
#         predicates = []
#         for arg, info in role_assignments.items():
#             predicates.append(f"{info['role']}(e, {info['word']})")
#
#         if predicates:
#             return f"∃e.{frame_name}(e) ∧ {' ∧ '.join(predicates)}"
#         return f"∃e.{frame_name}(e)"
#
#     def _calculate_entropy_profile(self, words: List[str]) -> List[float]:
#         """Calculate entropy profile for next token prediction."""
#         entropy_profile = []
#
#         for i in range(len(words) - 1):
#             current_word = words[i]
#             following_counts = Counter()
#             total_count = 0
#
#             for bg, count in self.bigram_counts.items():
#                 if bg[0] == current_word:
#                     following_counts[bg[1]] += count
#                     total_count += count
#
#             if total_count > 0:
#                 probs = [count / total_count for count in following_counts.values()]
#                 entropy = -sum(p * math.log2(p) for p in probs if p > 0)
#
#                 # Adjust toward medium range
#                 if entropy < 1.0:
#                     entropy = 1.0 + (entropy * 0.5)
#                 elif entropy > 3.0:
#                     entropy = 3.0 - ((entropy - 3.0) * 0.5)
#
#                 entropy_profile.append(entropy)
#             else:
#                 entropy_profile.append(2.0)  # Medium predictability
#
#         entropy_profile.append(2.8)  # End of sentence
#         return entropy_profile
#
#     def get_predictability_distribution(self, entropy_profiles: List[List[float]]) -> Dict[str, float]:
#         """Calculate predictability distribution across corpus."""
#         entropy_values = []
#         for profile in entropy_profiles:
#             entropy_values.extend(profile)
#
#         if not entropy_values:
#             return {"high_predictability": 0, "medium_predictability": 0, "low_predictability": 0}
#
#         total = len(entropy_values)
#         high_pred = sum(1 for e in entropy_values if e < 1.5) / total
#         medium_pred = sum(1 for e in entropy_values if 1.5 <= e < 3.0) / total
#         low_pred = sum(1 for e in entropy_values if e >= 3.0) / total
#
#         return {
#             "high_predictability": high_pred,
#             "medium_predictability": medium_pred,
#             "low_predictability": low_pred
#         }
# import random
# import math
# from collections import Counter
# from typing import Dict, List, Optional, Tuple, Any
#
# from ..lexicon.semantic_roles import SemanticRole
# from .linguistic_annotator import LinguisticAnnotator
#
#
# class SentenceGenerator:
#     """
#     Generates sentences with semantic frame structure and rich linguistic annotations.
#     """
#
#     def __init__(self, lexicon_generator, template_manager):
#         """
#         Initialize sentence generator.
#
#         Args:
#             lexicon_generator: Instance of LexiconGenerator
#             template_manager: Instance of TemplateManager
#         """
#         self.lexicon = lexicon_generator
#         self.templates = template_manager
#         self.annotator = LinguisticAnnotator()
#
#         # Statistics tracking
#         self.bigram_counts = Counter()
#         self.sentence_lengths = []
#
#     def _expand_frame_argument(self, arg: str, role: SemanticRole, frame_name: str, context: Dict) -> str:
#         """
#         Expand a frame argument (arg0, arg1, etc.) into an actual word based on semantic role.
#
#         Args:
#             arg: Argument identifier (arg0, arg1, etc.)
#             role: Semantic role for this argument
#             frame_name: Name of the semantic frame
#             context: Generation context
#
#         Returns:
#             Generated word for this argument
#         """
#         # Get semantic frame for role-to-POS mapping
#         frame = self.lexicon.get_semantic_frame(frame_name)
#         if frame and role in frame.pos_mapping:
#             pos_category = frame.pos_mapping[role]
#         else:
#             # Fallback mapping
#             pos_mapping = {
#                 SemanticRole.AGENT: "noun",
#                 SemanticRole.PATIENT: "noun",
#                 SemanticRole.THEME: "noun",
#                 SemanticRole.EXPERIENCER: "noun",
#                 SemanticRole.INSTRUMENT: "noun",
#                 SemanticRole.LOCATION: "location",
#                 SemanticRole.SOURCE: "location",
#                 SemanticRole.GOAL: "location",
#                 SemanticRole.TIME: "temporal"
#             }
#             pos_category = pos_mapping.get(role, "noun")
#
#         return self.lexicon.sample_word(pos_category, context)
#
#     def _expand_template_element(self, element: str, frame_name: str, roles: Dict, context: Dict) -> str:
#         """
#         Expand any template element (arguments, verbs, prepositions, etc.).
#
#         Args:
#             element: Template element to expand
#             frame_name: Semantic frame name
#             roles: Role mappings for arguments
#             context: Generation context
#
#         Returns:
#             Expanded word
#         """
#         # Handle frame arguments (arg0, arg1, etc.)
#         if element in roles:
#             return self._expand_frame_argument(element, roles[element], frame_name, context)
#
#         # Handle special argument types
#         if element == "arg_time":
#             return self.lexicon.sample_word("temporal", context)
#         elif element == "arg_instr":
#             return self.lexicon.sample_word("noun", context)  # Instrument is typically a noun
#
#         # Handle verbs
#         if element == "verb":
#             verb_mapping = {
#                 "transitive_action": "transitive_verb",
#                 "intransitive_action": "intransitive_verb",
#                 "communication": "communication_verb",
#                 "motion": "motion_verb"
#             }
#             verb_type = verb_mapping.get(frame_name, "transitive_verb")
#             word = self.lexicon.sample_word(verb_type, context)
#
#             # Apply tense
#             if context.get("tense") == "past":
#                 if random.random() < 0.1:  # 10% irregular verbs
#                     return f"{word}{random.choice(['t', 'ed', 'ew', 'ook'])}"
#                 return f"{word}ed"
#             else:
#                 return f"{word}s" if random.random() > 0.1 else word
#
#         elif element in ["verb1", "verb2"]:
#             # For multi-verb templates, use frame-appropriate verbs
#             return self._expand_template_element("verb", frame_name, roles, context)
#
#         # Handle other elements
#         element_mapping = {
#             "adj": "adjective",
#             "adv": "adverb",
#             "prep": "preposition",
#             "conj": "conjunction"
#         }
#
#         if element in element_mapping:
#             return self.lexicon.sample_word(element_mapping[element], context)
#
#         # Fallback
#         return self.lexicon.sample_word(element, context)
#
#     def generate_sentence(self, complexity: Optional[str] = None, include_metadata: bool = True) -> Dict[str, Any]:
#         """
#         Generate a sentence with semantic frame structure and linguistic annotations.
#
#         Args:
#             complexity: Optional complexity level
#             include_metadata: Whether to include metadata
#
#         Returns:
#             Dictionary with sentence, semantic roles, and linguistic annotations
#         """
#         # Select template
#         if complexity:
#             weights = {c: 1.0 if c == complexity else 0.0 for c in ["simple", "medium", "complex"]}
#             template_dict, complexity_level = self.templates.select_template(weights)
#         else:
#             template_dict, complexity_level = self.templates.select_template()
#
#         # Extract template information
#         frame_name = template_dict["frame"]
#         args_sequence = template_dict["args"]
#         roles = template_dict["roles"]
#
#         # Generation context
#         context = {"tense": "present", "frame": frame_name}
#
#         # Generate words and track semantic roles
#         words = []
#         semantic_annotations = {}
#         role_assignments = {}
#
#         for i, element in enumerate(args_sequence):
#             word = self._expand_template_element(element, frame_name, roles, context)
#             words.append(word)
#
#             # Track semantic roles for arguments
#             if element in roles:
#                 role_assignments[element] = {
#                     "word": word,
#                     "role": roles[element].value,
#                     "position": i
#                 }
#
#             # Update bigram statistics
#             if i > 0:
#                 bigram = (words[i - 1], word)
#                 self.bigram_counts[bigram] += 1
#
#         # Create sentence
#         sentence = " ".join(words)
#         self.sentence_lengths.append(len(words))
#
#         # Generate linguistic annotations
#         annotations = self.annotator.annotate_sentence(
#             sentence=sentence,
#             words=words,
#             role_assignments=role_assignments,
#             frame_name=frame_name
#         )
#
#         # Create semantic representation
#         semantics = self._create_semantic_representation(role_assignments, frame_name)
#
#         # Calculate entropy profile
#         entropy_profile = self._calculate_entropy_profile(words)
#
#         result = {
#             "sentence": sentence,
#             "semantic_roles": role_assignments,
#             "semantics": semantics,
#             "linguistic_annotations": annotations
#         }
#
#         if include_metadata:
#             result["metadata"] = {
#                 "complexity": complexity_level,
#                 "frame": frame_name,
#                 "template": template_dict,
#                 "length": len(words),
#                 "entropy_profile": entropy_profile,
#                 "avg_entropy": sum(entropy_profile) / len(entropy_profile) if entropy_profile else 0
#             }
#
#         return result
#
#     def _create_semantic_representation(self, role_assignments: Dict, frame_name: str) -> str:
#         """Create formal semantic representation."""
#         predicates = []
#         for arg, info in role_assignments.items():
#             predicates.append(f"{info['role']}(e, {info['word']})")
#
#         if predicates:
#             return f"∃e.{frame_name}(e) ∧ {' ∧ '.join(predicates)}"
#         return f"∃e.{frame_name}(e)"
#
#     def _calculate_entropy_profile(self, words: List[str]) -> List[float]:
#         """Calculate entropy profile for next token prediction."""
#         entropy_profile = []
#
#         for i in range(len(words) - 1):
#             current_word = words[i]
#             following_counts = Counter()
#             total_count = 0
#
#             for bg, count in self.bigram_counts.items():
#                 if bg[0] == current_word:
#                     following_counts[bg[1]] += count
#                     total_count += count
#
#             if total_count > 0:
#                 probs = [count / total_count for count in following_counts.values()]
#                 entropy = -sum(p * math.log2(p) for p in probs if p > 0)
#
#                 # Adjust toward medium range
#                 if entropy < 1.0:
#                     entropy = 1.0 + (entropy * 0.5)
#                 elif entropy > 3.0:
#                     entropy = 3.0 - ((entropy - 3.0) * 0.5)
#
#                 entropy_profile.append(entropy)
#             else:
#                 entropy_profile.append(2.0)  # Medium predictability
#
#         entropy_profile.append(2.8)  # End of sentence
#         return entropy_profile
#
#     def get_predictability_distribution(self, entropy_profiles: List[List[float]]) -> Dict[str, float]:
#         """Calculate predictability distribution across corpus."""
#         entropy_values = []
#         for profile in entropy_profiles:
#             entropy_values.extend(profile)
#
#         if not entropy_values:
#             return {"high_predictability": 0, "medium_predictability": 0, "low_predictability": 0}
#
#         total = len(entropy_values)
#         high_pred = sum(1 for e in entropy_values if e < 1.5) / total
#         medium_pred = sum(1 for e in entropy_values if 1.5 <= e < 3.0) / total
#         low_pred = sum(1 for e in entropy_values if e >= 3.0) / total
#
#         return {
#             "high_predictability": high_pred,
#             "medium_predictability": medium_pred,
#             "low_predictability": low_pred
#         }

# import random
# import math
# from collections import Counter
# from ..lexicon import LexiconGenerator
#
#
# class SentenceGenerator:
#     """
#     Generates sentences with controlled linguistic properties
#     for next token prediction tasks.
#     """
#
#     def __init__(self, lexicon_generator, template_manager):
#         """
#         Initialize with lexicon and template components.
#
#         Args:
#             lexicon_generator: Instance of LexiconGenerator
#             template_manager: Instance of TemplateManager
#         """
#         self.lexicon = lexicon_generator
#         self.templates = template_manager
#         self.bigram_counts = Counter()
#         self.sentence_lengths = []
#
#     def _expand_template_tag(self, tag, context=None):
#         """
#         Expand a template tag into an actual word.
#
#         Args:
#             tag: Template tag (e.g., "NOUN", "ADJ", "VERB")
#             context: Optional context information
#
#         Returns:
#             The expanded word
#         """
#         # Map template tags to word categories
#         tag_mapping = {
#             "NOUN": "noun",
#             "VERB": random.choice(["transitive_verb", "intransitive_verb", "motion_verb", "communication_verb"]),
#             "TVERB": "transitive_verb",
#             "IVERB": "intransitive_verb",
#             "MVERB": "motion_verb",
#             "CVERB": "communication_verb",
#             "ADJ": "adjective",
#             "ADV": "adverb",
#             "PREP": "preposition",  # For regular prepositions
#             "LOCATION": "location",  # Explicit mapping for location category
#             "DET": "determiner",
#             "CONJ": "conjunction",
#             "REL": "rel",
#             "COMP": "comp",
#             "TEMP": "temporal",
#             "RESULT": "result"  # Added result category
#         }
#
#         category = tag_mapping.get(tag, tag)
#
#         word = self.lexicon.sample_word(category, context)
#
#         # Handle verb conjugation (slightly more varied)
#         if tag == "VERB" or tag == "TVERB" or tag == "IVERB" or tag == "MVERB" or tag == "CVERB":
#             if context and context.get("tense") == "past":
#                 # Add some irregularity to make more interesting patterns
#                 if random.random() < 0.1:  # 10% chance of irregular verb
#                     return f"{word}{random.choice(['t', 'ed', 'ew', 'ook'])}"
#                 return f"{word}ed"
#             else:
#                 # Present tense - occasionally use different endings
#                 if random.random() < 0.1:  # 10% chance of variation
#                     return word  # Base form (for modal constructions)
#                 return f"{word}s"
#
#         return word
#
#
#     def generate_sentence(self, complexity=None, include_metadata=True):
#         """
#         Generate a single sentence with controlled linguistic properties.
#
#         Args:
#             complexity: Optional specific complexity level ("simple", "medium", "complex")
#
#         Returns:
#             Dictionary with sentence, semantic representation, and metadata
#         """
#         # Select template based on complexity distribution
#         if complexity:
#             weights = {c: 1.0 if c == complexity else 0.0 for c in ["simple", "medium", "complex"]}
#             template, complexity_level = self.templates.select_template(weights)
#         else:
#             template, complexity_level = self.templates.select_template()
#
#         words = []
#         semantics_dict = {}
#         context = {"head": None, "tense": "present"}
#
#         # Generate sentence by expanding template tags
#         for i, tag in enumerate(template):
#             # Update context based on sentence structure
#             if i > 0 and tag == "NOUN" and template[i-1] == "DET":
#                 # Reset head when we start a new noun phrase
#                 context["head"] = None
#
#             # Expand tag to actual word
#             word = self._expand_template_tag(tag, context)
#             words.append(word)
#
#             # Update context for next words
#             if tag == "NOUN":
#                 context["head"] = word  # Noun becomes head for adjectives
#                 if "Agent" not in semantics_dict:
#                     semantics_dict["Agent"] = word
#                 elif "Patient" not in semantics_dict:
#                     semantics_dict["Patient"] = word
#             elif tag == "VERB" or tag == "TVERB":
#                 if "Predicate" not in semantics_dict:
#                     semantics_dict["Predicate"] = word
#                 context["head"] = word  # Verb becomes head for objects
#
#             # Update bigram statistics
#             if i > 0:
#                 bigram = (words[i-1], word)
#                 self.bigram_counts[bigram] += 1
#
#         # Format the sentence with proper spacing and capitalization
#         formatted_words = []
#         for i, word in enumerate(words):
#             if i > 0 and word == ",":
#                 # No space before comma
#                 formatted_words[-1] = formatted_words[-1] + word
#             elif i > 0 and words[i-1] == ",":
#                 # Space after comma
#                 formatted_words.append(word)
#             # elif i == 0:
#             #     # Capitalize first word
#             #     formatted_words.append(word.capitalize())
#             else:
#                 formatted_words.append(word)
#
#         sentence = " ".join(formatted_words)
#
#         # Record sentence length for statistics
#         self.sentence_lengths.append(len(words))
#
#         # Create simplified semantics representation
#         semantics = f"∃e ({' ∧ '.join([f'{k}(e, {v})' for k, v in semantics_dict.items()])})"
#
#         # Calculate entropy profile for this sentence
#         entropy_profile = self._calculate_entropy_profile(words)
#
#         # Create metadata with information useful for next token prediction evaluation
#         metadata = {
#             "complexity": complexity_level,
#             "template": str(template),
#             "length": len(words),
#             "entropy_profile": entropy_profile,
#             "avg_entropy": sum(entropy_profile) / len(entropy_profile) if entropy_profile else 0
#         }
#         if include_metadata:
#
#             result = {
#                 "sentence": sentence,
#                 "semantics": semantics,
#                 "metadata": metadata
#             }
#         else:
#             result = {
#                 "sentence": sentence,
#                 "semantics": semantics
#             }
#
#         return result
#
#     # In sentence_generator.py, modify the _calculate_entropy_profile method slightly:
#
#     def _calculate_entropy_profile(self, words):
#         """Calculate the entropy profile with more medium-predictability values."""
#         entropy_profile = []
#
#         # For each position, calculate entropy of potential next words
#         for i in range(len(words) - 1):
#             current_word = words[i]
#
#             # Find all instances of current_word and count following words
#             following_counts = Counter()
#             total_count = 0
#
#             for bg, count in self.bigram_counts.items():
#                 if bg[0] == current_word:
#                     following_counts[bg[1]] += count
#                     total_count += count
#
#             # Calculate entropy if we have seen this word before
#             if total_count > 0:
#                 probs = [count / total_count for count in following_counts.values()]
#                 entropy = -sum(p * math.log2(p) for p in probs if p > 0)
#
#                 # Adjust entropy toward the medium range if it's too extreme
#                 if entropy < 1.0:  # Too predictable
#                     entropy = 1.0 + (entropy * 0.5)  # Shift toward medium (1.0-2.5)
#                 elif entropy > 3.0:  # Too unpredictable
#                     entropy = 3.0 - ((entropy - 3.0) * 0.5)  # Shift toward medium
#
#                 entropy_profile.append(entropy)
#             else:
#                 # If we haven't seen this word, assign medium entropy
#                 entropy_profile.append(2.0)  # Medium predictability instead of 2.5
#
#         # For the last position, assign medium-high entropy
#         entropy_profile.append(2.8)  # Reduced from 3.5
#
#         return entropy_profile
#
#     def get_predictability_distribution(self, entropy_profiles):
#         """
#         Calculate the distribution of predictability levels in the corpus.
#
#         Args:
#             entropy_profiles: List of entropy profiles from generated sentences
#
#         Returns:
#             Dictionary with predictability distribution
#         """
#         entropy_values = []
#         for profile in entropy_profiles:
#             entropy_values.extend(profile)
#
#         if not entropy_values:
#             return {
#                 "high_predictability": 0,
#                 "medium_predictability": 0,
#                 "low_predictability": 0
#             }
#
#         # Categorize by predictability
#         high_pred = sum(1 for e in entropy_values if e < 1.5) / len(entropy_values)
#         medium_pred = sum(1 for e in entropy_values if 1.5 <= e < 3.0) / len(entropy_values)
#         low_pred = sum(1 for e in entropy_values if e >= 3.0) / len(entropy_values)
#
#         return {
#             "high_predictability": high_pred,
#             "medium_predictability": medium_pred,
#             "low_predictability": low_pred
#         }
#
#     def get_bigram_statistics(self):
#         """
#         Get statistics about bigram distribution in generated sentences.
#
#         Returns:
#             Dictionary with bigram statistics
#         """
#         total_bigrams = sum(self.bigram_counts.values())
#         unique_bigrams = len(self.bigram_counts)
#
#         bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
#
#         # Top bigrams with counts
#         top_bigrams = self.bigram_counts.most_common(20)
#
#         # Calculate mutual information for common bigrams
#         mutual_info = {}
#         word_counts = Counter()
#
#         # Count individual words
#         for (w1, w2), count in self.bigram_counts.items():
#             word_counts[w1] += count
#             word_counts[w2] += count
#
#         # Calculate mutual information
#         for (w1, w2), joint_prob in self.bigram_counts.most_common(50):
#             joint_prob = joint_prob / total_bigrams
#             w1_prob = word_counts[w1] / (total_bigrams + len(self.sentence_lengths))
#             w2_prob = word_counts[w2] / (total_bigrams + len(self.sentence_lengths))
#
#             # Mutual information formula: log2(P(w1,w2) / (P(w1) * P(w2)))
#             if joint_prob > 0 and w1_prob > 0 and w2_prob > 0:
#                 mi = math.log2(joint_prob / (w1_prob * w2_prob))
#                 mutual_info[(w1, w2)] = mi
#
#         return {
#             "total_bigrams": total_bigrams,
#             "unique_bigrams": unique_bigrams,
#             "bigram_diversity": bigram_diversity,
#             "top_bigrams": top_bigrams,
#             "mutual_information": mutual_info
#         }