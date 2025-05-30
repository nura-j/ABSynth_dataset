import random


class TemplateManager:
    """
    Manages sentence templates with varying complexity for
    generating linguistically diverse synthetic sentences.
    """

    def __init__(self):
        """Initialize template manager with balanced templates."""
        self.templates = self._create_balanced_templates()
        self.template_usage = {}
        self.complexity_distribution = {"simple": 0, "medium": 0, "complex": 0}

        # Initialize template usage counters
        for complexity, templates_dict in self.templates.items():
            for template in templates_dict:
                self.template_usage[str(template)] = 0

    def _create_balanced_templates(self):
        """Create sentence templates with balanced POS distribution without COMP and REL."""
        return {
            # Simple patterns (target: 55% of corpus)
            "simple": {
                # Basic patterns with varied verb types and fewer DETs
                ("NOUN", "TRANSITIVE_VERB", "NOUN"): 0.12,
                ("NOUN", "INTRANSITIVE_VERB"): 0.12,
                ("NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.12,
                ("NOUN", "COMMUNICATION_VERB", "RESULT"): 0.12,

                # Modifier patterns
                ("ADJ", "NOUN", "TRANSITIVE_VERB", "NOUN"): 0.12,
                ("NOUN", "INTRANSITIVE_VERB", "ADV"): 0.12,

                # Location patterns
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "LOCATION"): 0.14,

                # Object modification
                ("NOUN", "TRANSITIVE_VERB", "ADJ", "NOUN"): 0.14
            },

            # Medium patterns (target: 35% of corpus)
            "medium": {
                # Prepositional phrases
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "NOUN"): 0.15,
                ("NOUN", "MOTION_VERB", "PREP", "LOCATION", "PREP", "NOUN"): 0.15,

                # Coordination
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "CONJ", "NOUN"): 0.14,

                # Temporal patterns
                ("TEMPORAL", "NOUN", "COMMUNICATION_VERB", "RESULT"): 0.14,

                # Complex noun phrases
                ("ADJ", "ADJ", "NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.14,

                # Adverbial patterns
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "TEMPORAL", "NOUN", "INTRANSITIVE_VERB"): 0.14,

                # Modified objects
                ("NOUN", "COMMUNICATION_VERB", "ADJ", "RESULT", "CONJ", "ADJ", "RESULT"): 0.14
            },

            # Complex patterns (target: 10% of corpus)
            "complex": {
                # Multi-verb patterns
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.25,

                # Multiple clauses
                ("CONJ", "NOUN", "MOTION_VERB", "PREP", "LOCATION", "NOUN", "CHANGE_VERB", "RESULT"): 0.25,

                # Complex temporal structures
                ("TEMPORAL", "NOUN", "TRANSITIVE_VERB", "NOUN", "NOUN", "COMMUNICATION_VERB", "RESULT"): 0.25,

                # Mixed complex structures
                ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "LOCATION", "CONJ", "INTRANSITIVE_VERB", "ADV"): 0.25
            }
        }
    # def _create_balanced_templates(self):
    #     """Create sentence templates with balanced POS distribution and linguistic realism."""
    #     return {
    #         # Simple patterns (target: 55% of corpus)
    #         "simple": {
    #             # Basic patterns with varied verb types
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN"): 0.1,
    #             ("DET", "NOUN", "INTRANSITIVE_VERB"): 0.1,
    #             ("DET", "NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.1,  # Motion verbs with locations
    #             ("DET", "NOUN", "COMMUNICATION_VERB", "COMP", "RESULT"): 0.1,  # Communication with results
    #
    #             # Modifier patterns
    #             ("DET", "ADJ", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN"): 0.1,
    #             ("DET", "NOUN", "INTRANSITIVE_VERB", "ADV"): 0.1,
    #
    #             # Location and result patterns
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN", "PREP", "LOCATION"): 0.1,
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "RESULT", "PREP", "DET", "NOUN"): 0.1,
    #
    #             # Object modification
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "DET", "ADJ", "NOUN"): 0.1
    #         },
    #
    #         # Medium patterns (target: 35% of corpus)
    #         "medium": {
    #             # Prepositional phrases
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN", "PREP", "DET", "NOUN"): 0.1,
    #             ("DET", "NOUN", "MOTION_VERB", "PREP", "LOCATION", "PREP", "DET", "NOUN"): 0.1,
    #
    #             # Coordination
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"): 0.1,
    #
    #             # Temporal patterns
    #             ("TEMP", "DET", "NOUN", "COMMUNICATION_VERB", "COMP", "RESULT"): 0.1,
    #
    #             # Complex noun phrases
    #             ("DET", "ADJ", "ADJ", "NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.1,
    #
    #             # Adverbial clauses
    #             ("DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN", "TEMP", "DET", "NOUN", "INTRANSITIVE_VERB"): 0.1,
    #
    #             # Modified objects with coordination
    #             ("DET", "NOUN", "COMMUNICATION_VERB", "DET", "ADJ", "RESULT", "CONJ", "DET", "ADJ", "RESULT"): 0.1,
    #
    #             # Complex verb phrases
    #             ("DET", "NOUN", "CHANGE_VERB", "ADV", "PREP", "DET", "NOUN"): 0.1
    #         },
    #
    #         # Complex patterns (target: 10% of corpus)
    #         "complex": {
    #             # Relative clauses with different verb types
    #             ("DET", "NOUN", "REL", "TRANSITIVE_VERB", "DET", "NOUN", "CHANGE_VERB", "RESULT"): 0.15,
    #             ("DET", "NOUN", "REL", "MOTION_VERB", "PREP", "LOCATION", "INTRANSITIVE_VERB", "ADV"): 0.15,
    #
    #             # Complement clauses
    #             ("DET", "NOUN", "COMMUNICATION_VERB", "COMP", "DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN"): 0.15,
    #
    #             # Multiple clauses with location and result
    #             ("CONJ", "DET", "NOUN", "MOTION_VERB", "PREP", "LOCATION", "COMMA", "DET", "NOUN", "CHANGE_VERB",
    #              "RESULT"): 0.15,
    #
    #             # Conditional structures
    #             ("TEMP", "DET", "NOUN", "TRANSITIVE_VERB", "DET", "NOUN", "COMMA", "DET", "NOUN", "COMMUNICATION_VERB",
    #              "RESULT"): 0.15,
    #
    #             # Embedded clauses
    #             ("DET", "NOUN", "COMMUNICATION_VERB", "COMP", "DET", "NOUN", "REL", "MOTION_VERB", "PREP", "LOCATION",
    #              "CHANGE_VERB", "RESULT"): 0.25
    #         }
    #     }
    # def _create_balanced_templates(self):
    #     """Create sentence templates with varying complexity and linguistic structures."""
    #     return {
    #         # Simple patterns (target: 55% of corpus)
    #         "simple": {
    #             # SV: The noun verbs
    #             ("DET", "NOUN", "VERB"): 0.15,
    #             # SVO: The noun verbs the noun
    #             ("DET", "NOUN", "VERB", "DET", "NOUN"): 0.25,  # Reduced from 0.3
    #             # SVO with adjective: The adj noun verbs the noun
    #             ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"): 0.2,  # Reduced from 0.25
    #             # SV with adverb: The noun verbs adv
    #             ("DET", "NOUN", "VERB", "ADV"): 0.15,  # Reduced from 0.2
    #             # SVO with adjective on object: The noun verbs the adj noun
    #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"): 0.1,
    #             # New: SVO with location: The noun verbs the noun at/in/on location
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "LOCATION"): 0.15  # Added for location
    #         },
    #
    #         # Medium patterns (target: 35% of corpus)
    #         "medium": {
    #             # SVO with PP: The noun verbs the noun prep the noun
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"): 0.15,  # Reduced from 0.2
    #             # SVO with coordination: The noun verbs the noun and the noun
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"): 0.1,  # Reduced from 0.15
    #             # SVO with temporal: Temp the noun verbs the noun
    #             ("TEMP", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.15,
    #             # Complex NP: The adj adj noun verbs prep the noun
    #             ("DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"): 0.1,  # Reduced from 0.15
    #             # SVO with adverbial clause: The noun verbs the noun when the noun verbs
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "TEMP", "DET", "NOUN", "VERB"): 0.15,
    #             # SVO with both objects modified: The noun verbs the adj noun and the adj noun
    #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "CONJ", "DET", "ADJ", "NOUN"): 0.1,
    #             # SVA with prepositional phrase: The noun verbs adv prep the noun
    #             ("DET", "NOUN", "VERB", "ADV", "PREP", "DET", "NOUN"): 0.1,
    #             # New: SVO with location: The noun verbs the noun at location
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "LOCATION"): 0.15  # Added for location
    #         },
    #
    #         # Complex patterns (target: 10% of corpus)
    #         "complex": {
    #             # Relative clause: The noun that verbs the noun verbs the noun
    #             ("DET", "NOUN", "REL", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.3,  # Reduced from 0.4
    #             # Complement clause: The noun verbs that the noun verbs the noun
    #             ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.3,
    #             # Multiple clauses: Conj the noun verbs, the noun verbs the noun
    #             ("CONJ", "DET", "NOUN", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.2,  # Reduced from 0.3
    #             # New: Complex with location: The noun that verbs at location verbs the noun
    #             ("DET", "NOUN", "REL", "VERB", "PREP", "LOCATION", "VERB", "DET", "NOUN"): 0.2  # Added for location
    #         }
    #     }
    # def _create_balanced_templates(self):
    #     """Create sentence templates with varying complexity and linguistic structures."""
    #     return {
    #         # Simple patterns (target: 55% of corpus)
    #         "simple": {
    #             # SV: The noun verbs
    #             ("DET", "NOUN", "VERB"): 0.15,
    #
    #             # SVO: The noun verbs the noun
    #             ("DET", "NOUN", "VERB", "DET", "NOUN"): 0.3,  # Reduced from 0.4
    #
    #             # SVO with adjective: The adj noun verbs the noun
    #             ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"): 0.25,
    #
    #             # SV with adverb: The noun verbs adv
    #             ("DET", "NOUN", "VERB", "ADV"): 0.2,
    #
    #             # New: SVO with adjective on object: The noun verbs the adj noun
    #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"): 0.1  # Added for variety
    #         },
    #
    #         # Medium patterns (target: 35% of corpus)
    #         "medium": {
    #             # SVO with PP: The noun verbs the noun prep the noun
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"): 0.2,  # Reduced from 0.3
    #
    #             # SVO with coordination: The noun verbs the noun and the noun
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"): 0.15,  # Reduced from 0.25
    #
    #             # SVO with temporal: Temp the noun verbs the noun
    #             ("TEMP", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.15,  # Reduced from 0.2
    #
    #             # Complex NP: The adj adj noun verbs prep the noun
    #             ("DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"): 0.15,  # Reduced from 0.25
    #
    #             # New: SVO with adverbial clause: The noun verbs the noun when the noun verbs
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "TEMP", "DET", "NOUN", "VERB"): 0.15,  # Added
    #
    #             # New: SVO with both objects modified: The noun verbs the adj noun and the adj noun
    #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "CONJ", "DET", "ADJ", "NOUN"): 0.1,  # Added
    #
    #             # New: SVA with prepositional phrase: The noun verbs adv prep the noun
    #             ("DET", "NOUN", "VERB", "ADV", "PREP", "DET", "NOUN"): 0.1  # Added
    #         },
    #
    #         # Complex patterns (target: 10% of corpus)
    #         "complex": {
    #             # Relative clause: The noun that verbs the noun verbs the noun
    #             ("DET", "NOUN", "REL", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.4,
    #
    #             # Complement clause: The noun verbs that the noun verbs the noun
    #             ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.3,
    #
    #             # Multiple clauses: Conj the noun verbs, the noun verbs the noun
    #             ("CONJ", "DET", "NOUN", "VERB", "COMMA", "DET", "NOUN", "VERB", "DET", "NOUN"): 0.3
    #         }
    #     }

    def select_template(self, complexity_weights=None):
        """
        Select a template based on target complexity distribution.

        Args:
            complexity_weights: Optional dictionary with weights for each complexity level
                e.g., {"simple": 0.55, "medium": 0.35, "complex": 0.1}

        Returns:
            Tuple of (template, complexity_level)
        """
        # Default complexity weights if not provided
        if complexity_weights is None:
            complexity_weights = {"simple": 0.55, "medium": 0.35, "complex": 0.1}

        # Select complexity level
        complexity_levels = list(complexity_weights.keys())
        complexity_weight_values = list(complexity_weights.values())

        # Safety check for non-empty weights
        if not complexity_weight_values or sum(complexity_weight_values) <= 0:
            # Fallback to default weights
            complexity_levels = ["simple", "medium", "complex"]
            complexity_weight_values = [0.55, 0.35, 0.1]

        complexity_level = random.choices(
            complexity_levels,
            weights=complexity_weight_values
        )[0]

        # Select template from the chosen complexity level
        template_dict = self.templates.get(complexity_level, self.templates["simple"])
        templates = list(template_dict.keys())
        weights = list(template_dict.values())

        # Safety check for non-empty templates and weights
        if not templates or not weights:
            # Fallback to a simple template
            template = ("NOUN", "VERB")
            complexity_level = "simple"
        else:
            template = random.choices(templates, weights=weights)[0]

        # Update stats
        self.complexity_distribution[complexity_level] += 1
        self.template_usage[str(template)] += 1

        return template, complexity_level

    def get_complexity_distribution(self, as_proportions=True):
        """
        Get the distribution of template complexities used.

        Args:
            as_proportions: Whether to return values as proportions (True) or counts (False)

        Returns:
            Dictionary with complexity distribution
        """
        total = sum(self.complexity_distribution.values())

        if as_proportions and total > 0:
            return {k: v / total for k, v in self.complexity_distribution.items()}
        else:
            return self.complexity_distribution.copy()

    def get_template_usage(self, top_n=None):
        """
        Get template usage statistics.

        Args:
            top_n: Optional number of top templates to return

        Returns:
            Dictionary mapping template strings to usage counts
        """
        if top_n:
            # Return top N most used templates
            sorted_templates = sorted(self.template_usage.items(),
                                      key=lambda x: x[1], reverse=True)
            return dict(sorted_templates[:top_n])
        else:
            return self.template_usage.copy()

    def add_custom_template(self, complexity, template, weight=0.2):
        """
        Add a custom template for sentence generation.

        Args:
            complexity: Complexity level ("simple", "medium", "complex")
            template: Tuple of POS tags defining the template
            weight: Relative weight for this template within its complexity level

        Returns:
            Boolean indicating success
        """
        if complexity not in self.templates:
            return False

        # Normalize existing weights to account for new template
        current_sum = sum(self.templates[complexity].values())
        adjustment_factor = (1.0 - weight) / current_sum if current_sum > 0 else 1.0

        for temp, w in self.templates[complexity].items():
            self.templates[complexity][temp] = w * adjustment_factor

        # Add new template
        self.templates[complexity][template] = weight
        self.template_usage[str(template)] = 0

        return True

    # def get_template_examples(self):
    #     """
    #     Get human-readable examples of each template type.
    #
    #     Returns:
    #         Dictionary with template examples
    #     """
    #     examples = {
    #         "simple": {
    #             ("DET", "NOUN", "VERB"): "The cat runs.",
    #             ("DET", "NOUN", "VERB", "DET", "NOUN"): "The dog chases the ball.",
    #             ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"): "The big man carries the box.",
    #             ("DET", "NOUN", "VERB", "ADV"): "The woman sings beautifully.",
    #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"): "The boy found the red book."
    #         },
    #         "medium": {
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"):
    #                 "The man put the book on the table.",
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"):
    #                 "The chef cooked the pasta and the sauce.",
    #             ("TEMP", "DET", "NOUN", "VERB", "DET", "NOUN"):
    #                 "Yesterday the teacher graded the papers.",
    #             ("DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"):
    #                 "The tall old building stands near the river.",
    #             ("DET", "NOUN", "VERB", "DET", "NOUN", "TEMP", "DET", "NOUN", "VERB"):
    #                 "The girl read the book while the boy played.",
    #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "CONJ", "DET", "ADJ", "NOUN"):
    #                 "The chef prepared the delicious pasta and the fresh salad.",
    #             ("DET", "NOUN", "VERB", "ADV", "PREP", "DET", "NOUN"):
    #                 "The child ran quickly toward the playground."
    #         },
    #         "complex": {
    #             ("DET", "NOUN", "REL", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"):
    #                 "The dog that chased the cat knocked over the lamp.",
    #             ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN", "VERB", "DET", "NOUN"):
    #                 "The woman said that the boy broke the window.",
    #             ("CONJ", "DET", "NOUN", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"):
    #                 "And the sun rose, the farmer harvested the crop."
    #         }
    #     }
    #
    #     return examples