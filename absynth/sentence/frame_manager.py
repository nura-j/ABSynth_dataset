import random
from typing import Dict, List, Tuple, Optional
from ..lexicon.semantic_roles import SemanticRole, SemanticFrame, SemanticRoles


class FrameManager:
    """
    Manages semantic frame templates with arg0, arg1, etc. structure
    for generating linguistically diverse synthetic sentences.
    """

    def __init__(self, frames: Optional[List[SemanticFrame]] = None):
        """Initialize template manager with semantic frame templates."""
        # Get semantic frames from lexicon module
        self.semantic_frames = SemanticRoles.get_standard_frames()

        # Create template dictionary with weights
        if frames is None:
            # Use default templates with weights
            # self.frames = self._create_weighted_templates()
            self.frames = self.create_weighted_templates()
        # elif isinstance(frames, dict):
        #     # check if each frame has a weight and a complexity level
        #     print('frames is a dict', frames)
        #     for frame in frames:
        #         if 'complexity' not in frames[frame] or 'weight' not in frames[frame]:
        #             raise ValueError("If frames is a dict, each frame must have 'weight' and 'complexity' keys.")
        #     # if not all("weight" in frame and "complexity" in frame for frame in frames.values()):
        #     #     raise ValueError("If frames is a dict, each frame must have 'weight' and 'complexity' keys.")
        #     self.templates = frames
        else:
            # User provided list of semantic frames - convert to weighted templates
            # print("Converting provided frames to templates...")
            self.frames = self._convert_frames_to_templates(frames)

        self.template_usage = {} # Track usage of each template
        self.complexity_distribution = {"simple": 0, "medium": 0, "complex": 0}

        # Initialize usage counters
        for complexity, templates_dict in self.frames.items():
            for template_name in templates_dict:
                self.template_usage[template_name] = 0

        # print("Initialized FrameManager with templates:")
        # print(self.templates)

    @staticmethod
    def _derive_args_from_frame(frame: SemanticFrame) -> List[str]:
        """Derive template args structure from semantic frame."""
        args = []
        for i, role in enumerate(frame.core_roles):
            args.append(f"arg{i}")
            if i < len(frame.core_roles) - 1:
                # Add connectors between arguments when appropriate
                args.append("verb" if i == 0 else "prep")
        return args

    @staticmethod
    def _derive_roles_from_frame(frame: SemanticFrame) -> Dict[str, SemanticRole]:
        """Map argument positions to semantic roles from frame."""
        role_mapping = {}
        for i, role in enumerate(frame.core_roles):
            role_mapping[f"arg{i}"] = role
        return role_mapping

    def _convert_frames_to_templates(self, frames: List[SemanticFrame], weight_method: str = 'distribution') -> Dict:
        """Convert semantic frames to weighted templates format."""
        templates = {"simple": {}, "medium": {}, "complex": {}}
        templates_weights = {'simple': 0.1, 'medium': 0.1, 'complex': 0.1}

        # Assign each frame to a template with default weight
        # We perform a basic complexity analysis based on the number of roles
        for frame in frames:
            # Default complexity based on number of roles
            complexity = "simple"
            if len(frames[frame].core_roles) >= 3:
                complexity = "medium"
            if len(frames[frame].core_roles) >= 6 or len(frames[frame].optional_roles) >= 3:
                complexity = "complex"
            flag = False
            template_name = f"{frames[frame].frame_name}"
            if frames[frame].weight:
                weight = frames[frame].weight
            elif weight_method == 'uniformal':
                weight = 0.1
            elif weight_method == 'random':
                weight = random.uniform(0.1, 0.3)
            else:  # Default to distribution method
                flag = True
            templates[complexity][template_name] = {
                "frame": frames[frame].frame_name,
                "args": self._derive_args_from_frame(frames[frame]),
                "roles": self._derive_roles_from_frame(frames[frame]),
                "weight": weight
            }

            if weight_method == 'distribution' or flag:
                for complexity in templates:
                    templates_weights[complexity] = 1 / len(templates[complexity]) if templates[complexity] else 0
                for complexity in templates:
                    for template in templates[complexity]:
                        templates[complexity][template]['weight'] = templates_weights[complexity]

        return templates

    @staticmethod
    def create_weighted_templates() -> Dict[str, Dict[str, Dict]]:
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
        template_dict = self.frames.get(complexity_level, self.frames["simple"])
        template_names = list(template_dict.keys())
        # print(f"Selected complexity level: {complexity_level}, available templates: {template_names}")
        weights = [template_dict[name]["weight"] for name in template_names]

        if not template_names or not weights:
            # Fallback
            template_name = "basic_transitive"
            template = self.frames["simple"]["basic_transitive"]
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

    def set_template_weights(self, weights: Dict[str, float]) -> None:
        """Set custom weights for templates."""
        for complexity in self.frames:
            for template_name in self.frames[complexity]:
                if template_name in weights:
                    self.frames[complexity][template_name]['weight'] = weights[template_name]

    def get_default_semantic_frames_distribution(self) -> Dict[str, float]:
        '''Get default distribution of template complexities.
        Returns:
            Dict[str, float]: Default distribution of template complexities.
        '''
        semantic_frames_distribution = {}
        default_frames = self.create_weighted_templates()
        for complexity in default_frames:
            for template_name in default_frames[complexity]:
                semantic_frames_distribution[template_name] = default_frames[complexity][template_name]['weight']

        return semantic_frames_distribution

