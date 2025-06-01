import random
from typing import Dict, List, Tuple, Optional
from ..lexicon.semantic_roles import SemanticRole, SemanticFrame


class TemplateManager:
    """
    Manages semantic frame templates with arg0, arg1, etc. structure
    for generating linguistically diverse synthetic sentences.
    """

    def __init__(self):
        """Initialize template manager with semantic frame templates."""
        self.templates = self._create_frame_templates()
        self.template_usage = {}
        self.complexity_distribution = {"simple": 0, "medium": 0, "complex": 0}

        # Initialize usage counters
        for complexity, templates_dict in self.templates.items():
            for template_name in templates_dict:
                self.template_usage[template_name] = 0

    def _create_frame_templates(self) -> Dict[str, Dict[str, Dict]]: #todo change to public method and allow user to change it and its weight or add new templates - each template shoule be a dataclass with type hints
        """
        Create semantic frame templates with standardized argument structure.
        Templates now use arg0, arg1, etc. with semantic role mappings.
        """
        return {
            # Simple frames (55% target)
            "simple": {
                "basic_transitive": {
                    "frame": "transitive_action",
                    "args": ["arg0", "verb", "arg1"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT
                    },
                    "weight": 0.25
                },

                "basic_intransitive": {
                    "frame": "intransitive_action",
                    "args": ["arg0", "verb"],
                    "roles": {
                        "arg0": SemanticRole.AGENT
                    },
                    "weight": 0.20
                },

                "motion_simple": {
                    "frame": "motion",
                    "args": ["arg0", "verb", "prep", "arg1"],
                    "roles": {
                        "arg0": SemanticRole.THEME,
                        "arg1": SemanticRole.GOAL
                    },
                    "weight": 0.15
                },

                "communication_simple": {
                    "frame": "communication",
                    "args": ["arg0", "verb", "arg1"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.THEME
                    },
                    "weight": 0.15
                },

                "modified_transitive": {
                    "frame": "transitive_action",
                    "args": ["adj", "arg0", "verb", "arg1"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT
                    },
                    "weight": 0.15
                },

                "adverbial_action": {
                    "frame": "intransitive_action",
                    "args": ["arg0", "verb", "adv"],
                    "roles": {
                        "arg0": SemanticRole.AGENT
                    },
                    "weight": 0.10
                }
            },

            # Medium complexity frames (35% target)
            "medium": {
                "transitive_with_location": {
                    "frame": "transitive_action",
                    "args": ["arg0", "verb", "arg1", "prep", "arg2"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT,
                        "arg2": SemanticRole.LOCATION
                    },
                    "weight": 0.20
                },

                "motion_with_source": {
                    "frame": "motion",
                    "args": ["arg0", "verb", "prep", "arg1", "prep", "arg2"],
                    "roles": {
                        "arg0": SemanticRole.THEME,
                        "arg1": SemanticRole.SOURCE,
                        "arg2": SemanticRole.GOAL
                    },
                    "weight": 0.15
                },

                "coordination": {
                    "frame": "transitive_action",
                    "args": ["arg0", "verb", "arg1", "conj", "arg2"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT,
                        "arg2": SemanticRole.PATIENT  # Second coordinated object
                    },
                    "weight": 0.15
                },

                "temporal_action": {
                    "frame": "transitive_action",
                    "args": ["arg_time", "arg0", "verb", "arg1"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT,
                        "arg_time": SemanticRole.TIME
                    },
                    "weight": 0.15
                },

                "communication_to_experiencer": {
                    "frame": "communication",
                    "args": ["arg0", "verb", "arg1", "prep", "arg2"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.THEME,
                        "arg2": SemanticRole.EXPERIENCER
                    },
                    "weight": 0.15
                },

                "instrumental_action": {
                    "frame": "transitive_action",
                    "args": ["arg0", "verb", "arg1", "prep", "arg_instr"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT,
                        "arg_instr": SemanticRole.INSTRUMENT
                    },
                    "weight": 0.20
                }
            },

            # Complex frames (10% target)
            "complex": {
                "multi_action": {
                    "frame": "transitive_action",
                    "args": ["arg0", "verb1", "arg1", "verb2", "prep", "arg2"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT,
                        "arg2": SemanticRole.GOAL
                    },
                    "weight": 0.30
                },

                "complex_coordination": {
                    "frame": "transitive_action",
                    "args": ["conj", "arg0", "verb1", "prep", "arg1", "arg2", "verb2", "arg3"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.LOCATION,
                        "arg2": SemanticRole.THEME,
                        "arg3": SemanticRole.PATIENT
                    },
                    "weight": 0.25
                },

                "temporal_complex": {
                    "frame": "communication",
                    "args": ["arg_time", "arg0", "verb1", "arg1", "arg2", "verb2", "arg3"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.THEME,
                        "arg2": SemanticRole.AGENT,  # Second clause agent
                        "arg3": SemanticRole.PATIENT,
                        "arg_time": SemanticRole.TIME
                    },
                    "weight": 0.25
                },

                "full_argument_structure": {
                    "frame": "transitive_action",
                    "args": ["arg0", "verb", "arg1", "prep", "arg2", "conj", "verb", "adv"],
                    "roles": {
                        "arg0": SemanticRole.AGENT,
                        "arg1": SemanticRole.PATIENT,
                        "arg2": SemanticRole.LOCATION
                    },
                    "weight": 0.20
                }
            }
        }

    def select_template(self, complexity_weights: Optional[Dict[str, float]] = None) -> Tuple[Dict, str]:
        """
        Select a semantic frame template based on complexity distribution.

        Args:
            complexity_weights: Optional weights for complexity levels

        Returns:
            Tuple of (template_dict, complexity_level)
        """
        if complexity_weights is None:
            complexity_weights = {"simple": 0.55, "medium": 0.35, "complex": 0.1}

        # Select complexity level
        complexity_levels = list(complexity_weights.keys())
        complexity_weight_values = list(complexity_weights.values())

        if not complexity_weight_values or sum(complexity_weight_values) <= 0:
            complexity_levels = ["simple", "medium", "complex"]
            complexity_weight_values = [0.55, 0.35, 0.1]

        complexity_level = random.choices(complexity_levels, weights=complexity_weight_values)[0]

        # Select template from chosen complexity
        template_dict = self.templates.get(complexity_level, self.templates["simple"])
        template_names = list(template_dict.keys())
        weights = [template_dict[name]["weight"] for name in template_names]

        if not template_names or not weights:
            # Fallback
            template_name = "basic_transitive"
            template = self.templates["simple"]["basic_transitive"]
            complexity_level = "simple"
        else:
            template_name = random.choices(template_names, weights=weights)[0]
            template = template_dict[template_name]

        # Update statistics
        self.complexity_distribution[complexity_level] += 1
        self.template_usage[template_name] += 1

        return template, complexity_level

    def get_complexity_distribution(self, as_proportions: bool = True) -> Dict[str, float]:
        """Get distribution of template complexities used."""
        total = sum(self.complexity_distribution.values())

        if as_proportions and total > 0:
            return {k: v / total for k, v in self.complexity_distribution.items()}
        return self.complexity_distribution.copy()

    def get_template_usage(self, top_n: Optional[int] = None) -> Dict[str, int]:
        """Get template usage statistics."""
        if top_n:
            sorted_templates = sorted(self.template_usage.items(),
                                      key=lambda x: x[1], reverse=True)
            return dict(sorted_templates[:top_n])
        return self.template_usage.copy()

# import random
# from typing import Dict, List, Tuple, Optional
# from ..lexicon.semantic_roles import SemanticRole, SemanticFrame
#
#
# class TemplateManager:
#     """
#     Manages semantic frame templates with arg0, arg1, etc. structure
#     for generating linguistically diverse synthetic sentences.
#     """
#
#     def __init__(self):
#         """Initialize template manager with semantic frame templates."""
#         self.templates = self._create_frame_templates()
#         self.template_usage = {}
#         self.complexity_distribution = {"simple": 0, "medium": 0, "complex": 0}
#
#         # Initialize usage counters
#         for complexity, templates_dict in self.templates.items():
#             for template_name in templates_dict:
#                 self.template_usage[template_name] = 0
#
#     def _create_frame_templates(self) -> Dict[str, Dict[str, Dict]]: #todo change to public method and allow user to change it and its weight or add new templates
#         """
#         Create semantic frame templates with standardized argument structure.
#         Templates now use arg0, arg1, etc. with semantic role mappings.
#         """
#         return {
#             # Simple frames (55% target)
#             "simple": {
#                 "basic_transitive": {
#                     "frame": "transitive_action",
#                     "args": ["arg0", "verb", "arg1"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT
#                     },
#                     "weight": 0.25
#                 },
#
#                 "basic_intransitive": {
#                     "frame": "intransitive_action",
#                     "args": ["arg0", "verb"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT
#                     },
#                     "weight": 0.20
#                 },
#
#                 "motion_simple": {
#                     "frame": "motion",
#                     "args": ["arg0", "verb", "prep", "arg1"],
#                     "roles": {
#                         "arg0": SemanticRole.THEME,
#                         "arg1": SemanticRole.GOAL
#                     },
#                     "weight": 0.15
#                 },
#
#                 "communication_simple": {
#                     "frame": "communication",
#                     "args": ["arg0", "verb", "arg1"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.THEME
#                     },
#                     "weight": 0.15
#                 },
#
#                 "modified_transitive": {
#                     "frame": "transitive_action",
#                     "args": ["adj", "arg0", "verb", "arg1"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT
#                     },
#                     "weight": 0.15
#                 },
#
#                 "adverbial_action": {
#                     "frame": "intransitive_action",
#                     "args": ["arg0", "verb", "adv"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT
#                     },
#                     "weight": 0.10
#                 }
#             },
#
#             # Medium complexity frames (35% target)
#             "medium": {
#                 "transitive_with_location": {
#                     "frame": "transitive_action",
#                     "args": ["arg0", "verb", "arg1", "prep", "arg2"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT,
#                         "arg2": SemanticRole.LOCATION
#                     },
#                     "weight": 0.20
#                 },
#
#                 "motion_with_source": {
#                     "frame": "motion",
#                     "args": ["arg0", "verb", "prep", "arg1", "prep", "arg2"],
#                     "roles": {
#                         "arg0": SemanticRole.THEME,
#                         "arg1": SemanticRole.SOURCE,
#                         "arg2": SemanticRole.GOAL
#                     },
#                     "weight": 0.15
#                 },
#
#                 "coordination": {
#                     "frame": "transitive_action",
#                     "args": ["arg0", "verb", "arg1", "conj", "arg2"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT,
#                         "arg2": SemanticRole.PATIENT  # Second coordinated object
#                     },
#                     "weight": 0.15
#                 },
#
#                 "temporal_action": {
#                     "frame": "transitive_action",
#                     "args": ["arg_time", "arg0", "verb", "arg1"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT,
#                         "arg_time": SemanticRole.TIME
#                     },
#                     "weight": 0.15
#                 },
#
#                 "communication_to_experiencer": {
#                     "frame": "communication",
#                     "args": ["arg0", "verb", "arg1", "prep", "arg2"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.THEME,
#                         "arg2": SemanticRole.EXPERIENCER
#                     },
#                     "weight": 0.15
#                 },
#
#                 "instrumental_action": {
#                     "frame": "transitive_action",
#                     "args": ["arg0", "verb", "arg1", "prep", "arg_instr"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT,
#                         "arg_instr": SemanticRole.INSTRUMENT
#                     },
#                     "weight": 0.20
#                 }
#             },
#
#             # Complex frames (10% target)
#             "complex": {
#                 "multi_action": {
#                     "frame": "transitive_action",
#                     "args": ["arg0", "verb1", "arg1", "verb2", "prep", "arg2"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT,
#                         "arg2": SemanticRole.GOAL
#                     },
#                     "weight": 0.30
#                 },
#
#                 "complex_coordination": {
#                     "frame": "transitive_action",
#                     "args": ["conj", "arg0", "verb1", "prep", "arg1", "arg2", "verb2", "arg3"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.LOCATION,
#                         "arg2": SemanticRole.THEME,
#                         "arg3": SemanticRole.PATIENT
#                     },
#                     "weight": 0.25
#                 },
#
#                 "temporal_complex": {
#                     "frame": "communication",
#                     "args": ["arg_time", "arg0", "verb1", "arg1", "arg2", "verb2", "arg3"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.THEME,
#                         "arg2": SemanticRole.AGENT,  # Second clause agent
#                         "arg3": SemanticRole.PATIENT,
#                         "arg_time": SemanticRole.TIME
#                     },
#                     "weight": 0.25
#                 },
#
#                 "full_argument_structure": {
#                     "frame": "transitive_action",
#                     "args": ["arg0", "verb", "arg1", "prep", "arg2", "conj", "verb", "adv"],
#                     "roles": {
#                         "arg0": SemanticRole.AGENT,
#                         "arg1": SemanticRole.PATIENT,
#                         "arg2": SemanticRole.LOCATION
#                     },
#                     "weight": 0.20
#                 }
#             }
#         }
#
#     def select_template(self, complexity_weights: Optional[Dict[str, float]] = None) -> Tuple[Dict, str]:
#         """
#         Select a semantic frame template based on complexity distribution.
#
#         Args:
#             complexity_weights: Optional weights for complexity levels
#
#         Returns:
#             Tuple of (template_dict, complexity_level)
#         """
#         if complexity_weights is None:
#             complexity_weights = {"simple": 0.55, "medium": 0.35, "complex": 0.1}
#
#         # Select complexity level
#         complexity_levels = list(complexity_weights.keys())
#         complexity_weight_values = list(complexity_weights.values())
#
#         if not complexity_weight_values or sum(complexity_weight_values) <= 0:
#             complexity_levels = ["simple", "medium", "complex"]
#             complexity_weight_values = [0.55, 0.35, 0.1]
#
#         complexity_level = random.choices(complexity_levels, weights=complexity_weight_values)[0]
#
#         # Select template from chosen complexity
#         template_dict = self.templates.get(complexity_level, self.templates["simple"])
#         template_names = list(template_dict.keys())
#         weights = [template_dict[name]["weight"] for name in template_names]
#
#         if not template_names or not weights:
#             # Fallback
#             template_name = "basic_transitive"
#             template = self.templates["simple"]["basic_transitive"]
#             complexity_level = "simple"
#         else:
#             template_name = random.choices(template_names, weights=weights)[0]
#             template = template_dict[template_name]
#
#         # Update statistics
#         self.complexity_distribution[complexity_level] += 1
#         self.template_usage[template_name] += 1
#
#         return template, complexity_level
#
#     def get_complexity_distribution(self, as_proportions: bool = True) -> Dict[str, float]:
#         """Get distribution of template complexities used."""
#         total = sum(self.complexity_distribution.values())
#
#         if as_proportions and total > 0:
#             return {k: v / total for k, v in self.complexity_distribution.items()}
#         return self.complexity_distribution.copy()
#
#     def get_template_usage(self, top_n: Optional[int] = None) -> Dict[str, int]:
#         """Get template usage statistics."""
#         if top_n:
#             sorted_templates = sorted(self.template_usage.items(),
#                                       key=lambda x: x[1], reverse=True)
#             return dict(sorted_templates[:top_n])
#         return self.template_usage.copy()

# import random
#
#
# class FrameManager:
#     """
#     Manages sentence templates with varying complexity for
#     generating linguistically diverse synthetic sentences.
#     """
#
#     def __init__(self):
#         """Initialize template manager with balanced templates."""
#         self.frames = self._create_balanced_frames()
#         self.frame_usage = {}
#         self.complexity_distribution = {"simple": 0, "medium": 0, "complex": 0}
#
#         # Initialize template usage counters
#         for complexity, frames_dict in self.frames.items():
#             for template in frames_dict:
#                 self.frame_usage[str(template)] = 0
#
#     def _create_balanced_frames(self): # this can be publis, expose it to the user and give them the freedom to chnage it, declare it before the class and if the user does not give options use it as default
#         """Create sentence templates with balanced POS distribution without COMP and REL."""
#         return {
#             # Simple patterns (target: 55% of corpus)
#             "simple": { # typed dict[[tuple(str), float]]:
#                 # Basic patterns with varied verb types and fewer DETs
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN"): 0.12,
#                 ("NOUN", "INTRANSITIVE_VERB"): 0.12,
#                 ("NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.12,
#                 ("NOUN", "COMMUNICATION_VERB", "RESULT"): 0.12,
#
#                 # Modifier patterns
#                 ("ADJ", "NOUN", "TRANSITIVE_VERB", "NOUN"): 0.12,
#                 ("NOUN", "INTRANSITIVE_VERB", "ADV"): 0.12,
#
#                 # Location patterns
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "LOCATION"): 0.14,
#
#                 # Object modification
#                 ("NOUN", "TRANSITIVE_VERB", "ADJ", "NOUN"): 0.14
#             },
#
#             # Medium patterns (target: 35% of corpus)
#             "medium": {
#                 # Prepositional phrases
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "NOUN"): 0.15,
#                 ("NOUN", "MOTION_VERB", "PREP", "LOCATION", "PREP", "NOUN"): 0.15,
#
#                 # Coordination
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN", "CONJ", "NOUN"): 0.14,
#
#                 # Temporal patterns
#                 ("TEMPORAL", "NOUN", "COMMUNICATION_VERB", "RESULT"): 0.14,
#
#                 # Complex noun phrases
#                 ("ADJ", "ADJ", "NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.14,
#
#                 # Adverbial patterns
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN", "TEMPORAL", "NOUN", "INTRANSITIVE_VERB"): 0.14,
#
#                 # Modified objects
#                 ("NOUN", "COMMUNICATION_VERB", "ADJ", "RESULT", "CONJ", "ADJ", "RESULT"): 0.14
#             },
#
#             # Complex patterns (target: 10% of corpus)
#             "complex": {
#                 # Multi-verb patterns
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN", "MOTION_VERB", "PREP", "LOCATION"): 0.25,
#
#                 # Multiple clauses
#                 ("CONJ", "NOUN", "MOTION_VERB", "PREP", "LOCATION", "NOUN", "CHANGE_VERB", "RESULT"): 0.25,
#
#                 # Complex temporal structures
#                 ("TEMPORAL", "NOUN", "TRANSITIVE_VERB", "NOUN", "NOUN", "COMMUNICATION_VERB", "RESULT"): 0.25,
#
#                 # Mixed complex structures
#                 ("NOUN", "TRANSITIVE_VERB", "NOUN", "PREP", "LOCATION", "CONJ", "INTRANSITIVE_VERB", "ADV"): 0.25
#             }
#         }
#
#     def select_template(self, complexity_weights=None):
#         """
#         Select a template based on target complexity distribution.
#
#         Args:
#             complexity_weights: Optional dictionary with weights for each complexity level
#                 e.g., {"simple": 0.55, "medium": 0.35, "complex": 0.1}
#
#         Returns:
#             Tuple of (template, complexity_level)
#         """
#         # Default complexity weights if not provided
#         if complexity_weights is None:
#             complexity_weights = {"simple": 0.55, "medium": 0.35, "complex": 0.1}
#
#         # Select complexity level
#         complexity_levels = list(complexity_weights.keys())
#         complexity_weight_values = list(complexity_weights.values())
#
#         # Safety check for non-empty weights
#         if not complexity_weight_values or sum(complexity_weight_values) <= 0:
#             # Fallback to default weights
#             complexity_levels = ["simple", "medium", "complex"]
#             complexity_weight_values = [0.55, 0.35, 0.1]
#
#         complexity_level = random.choices(
#             complexity_levels,
#             weights=complexity_weight_values
#         )[0]
#
#         # Select template from the chosen complexity level
#         template_dict = self.frames.get(complexity_level, self.frames["simple"])
#         templates = list(template_dict.keys())
#         weights = list(template_dict.values())
#
#         # Safety check for non-empty templates and weights
#         if not templates or not weights:
#             # Fallback to a simple template
#             template = ("NOUN", "VERB")
#             complexity_level = "simple"
#         else:
#             template = random.choices(templates, weights=weights)[0]
#
#         # Update stats
#         self.complexity_distribution[complexity_level] += 1
#         self.frame_usage[str(template)] += 1
#
#         return template, complexity_level
#
#     def get_complexity_distribution(self, as_proportions=True):
#         """
#         Get the distribution of template complexities used.
#
#         Args:
#             as_proportions: Whether to return values as proportions (True) or counts (False)
#
#         Returns:
#             Dictionary with complexity distribution
#         """
#         total = sum(self.complexity_distribution.values())
#
#         if as_proportions and total > 0:
#             return {k: v / total for k, v in self.complexity_distribution.items()}
#         else:
#             return self.complexity_distribution.copy()
#
#     def get_template_usage(self, top_n=None):
#         """
#         Get template usage statistics.
#
#         Args:
#             top_n: Optional number of top templates to return
#
#         Returns:
#             Dictionary mapping template strings to usage counts
#         """
#         if top_n:
#             # Return top N most used templates
#             sorted_templates = sorted(self.frame_usage.items(),
#                                       key=lambda x: x[1], reverse=True)
#             return dict(sorted_templates[:top_n])
#         else:
#             return self.frame_usage.copy()
#
#     def add_custom_template(self, complexity, template, weight=0.2):
#         """
#         Add a custom template for sentence generation.
#
#         Args:
#             complexity: Complexity level ("simple", "medium", "complex") - can be enum
#             template: Tuple of POS tags defining the template
#             weight: Relative weight for this template within its complexity level
#
#         Returns:
#             Boolean indicating success
#         """
#         if complexity not in self.frames:
#             return False
#
#         # Normalize existing weights to account for new template
#         current_sum = sum(self.frames[complexity].values())
#         adjustment_factor = (1.0 - weight) / current_sum if current_sum > 0 else 1.0
#
#         for temp, w in self.frames[complexity].items():
#             self.frames[complexity][temp] = w * adjustment_factor
#
#         # Add new template
#         self.frames[complexity][template] = weight
#         self.frame_usage[str(template)] = 0
#
#         return True
#
#     # def get_template_examples(self):
#     #     """
#     #     Get human-readable examples of each template type.
#     #
#     #     Returns:
#     #         Dictionary with template examples
#     #     """
#     #     examples = {
#     #         "simple": {
#     #             ("DET", "NOUN", "VERB"): "The cat runs.",
#     #             ("DET", "NOUN", "VERB", "DET", "NOUN"): "The dog chases the ball.",
#     #             ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"): "The big man carries the box.",
#     #             ("DET", "NOUN", "VERB", "ADV"): "The woman sings beautifully.",
#     #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"): "The boy found the red book."
#     #         },
#     #         "medium": {
#     #             ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"):
#     #                 "The man put the book on the table.",
#     #             ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"):
#     #                 "The chef cooked the pasta and the sauce.",
#     #             ("TEMP", "DET", "NOUN", "VERB", "DET", "NOUN"):
#     #                 "Yesterday the teacher graded the papers.",
#     #             ("DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"):
#     #                 "The tall old building stands near the river.",
#     #             ("DET", "NOUN", "VERB", "DET", "NOUN", "TEMP", "DET", "NOUN", "VERB"):
#     #                 "The girl read the book while the boy played.",
#     #             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "CONJ", "DET", "ADJ", "NOUN"):
#     #                 "The chef prepared the delicious pasta and the fresh salad.",
#     #             ("DET", "NOUN", "VERB", "ADV", "PREP", "DET", "NOUN"):
#     #                 "The child ran quickly toward the playground."
#     #         },
#     #         "complex": {
#     #             ("DET", "NOUN", "REL", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"):
#     #                 "The dog that chased the cat knocked over the lamp.",
#     #             ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN", "VERB", "DET", "NOUN"):
#     #                 "The woman said that the boy broke the window.",
#     #             ("CONJ", "DET", "NOUN", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN"):
#     #                 "And the sun rose, the farmer harvested the crop."
#     #         }
#     #     }
#     #
#     #     return examples