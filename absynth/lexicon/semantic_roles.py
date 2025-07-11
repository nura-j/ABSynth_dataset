from dataclasses import dataclass
from typing import Dict, List
from enum import StrEnum


class SemanticRole(StrEnum):
    """Standard semantic roles (thematic roles) for argument structure."""

    AGENT = "Agent"  # The doer of the action
    PATIENT = "Patient"  # The entity affected by the action
    THEME = "Theme"  # The entity moved or described
    EXPERIENCER = "Experiencer"  # The entity experiencing something
    INSTRUMENT = "Instrument"  # The tool used
    LOCATION = "Location"  # Where something happens
    SOURCE = "Source"  # Starting point
    GOAL = "Goal"  # End point
    TIME = "Time"  # When something happens
    # Stimulus Recipient Beneficiary Manner Purpose Cause


@dataclass
class SemanticFrame:
    """
    Represents a semantic frame with standardized argument structure.
    We define it as a dataclass to encapsulate the frame name, core roles, optional roles, and part-of-speech mappings -
    similar to out Vocabulary class.
    """
    frame_name: str
    core_roles: List[SemanticRole]
    optional_roles: List[SemanticRole] = None
    pos_mapping: Dict[SemanticRole, str] = None
    weight: float = None # Default weight for the frame

    def __post_init__(self):
        if self.optional_roles is None:
            self.optional_roles = []
        if self.pos_mapping is None:
            self.pos_mapping = {}


class SemanticRoles:
    """
    Manages semantic frames and role assignments for corpus generation.
    """
    @staticmethod
    def get_standard_frames() -> Dict[str, SemanticFrame]:
        """
        Returns standard semantic frames for common sentence types.
        """
        return {
            "transitive_action": SemanticFrame(
                frame_name="transitive_action",
                core_roles=[SemanticRole.AGENT, SemanticRole.PATIENT],
                optional_roles=[SemanticRole.INSTRUMENT, SemanticRole.LOCATION, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.PATIENT: "noun",
                        SemanticRole.INSTRUMENT: "noun",
                        SemanticRole.LOCATION: "location",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.2
            ),

            # "intransitive_action": SemanticFrame(
            #     frame_name="intransitive_action",
            #     core_roles=[SemanticRole.AGENT, ],
            #     optional_roles=[SemanticRole.LOCATION, SemanticRole.TIME],
            #     pos_mapping={
            #             SemanticRole.AGENT: "noun",
            #             SemanticRole.LOCATION: "location",
            #             SemanticRole.TIME: "temporal"
            #     },
            #     weight=0.20
            # ),
            "basic_transitive": SemanticFrame(
                frame_name="basic_transitive",
                core_roles=[SemanticRole.AGENT, SemanticRole.PATIENT],
                optional_roles=[],
                pos_mapping={
                       SemanticRole.AGENT: "noun",
                       SemanticRole.PATIENT: "noun"
                },
                weight=0.1
            ),
            "communication": SemanticFrame(
                frame_name="communication",
                core_roles=[SemanticRole.AGENT, SemanticRole.THEME],
                optional_roles=[SemanticRole.EXPERIENCER, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.THEME: "noun",
                        SemanticRole.EXPERIENCER: "noun",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.15
            ),
            "motion": SemanticFrame(
                frame_name="motion",
                core_roles=[SemanticRole.THEME,  SemanticRole.GOAL],
                optional_roles=[SemanticRole.SOURCE, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.THEME: "noun",
                        SemanticRole.SOURCE: "location",
                        SemanticRole.GOAL: "location",
                        SemanticRole.TIME: "temporal"
                },
                 weight=0.15
            ),
            "instrumental_action": SemanticFrame(
                frame_name="instrumental_action",
                core_roles=[SemanticRole.AGENT, SemanticRole.PATIENT, SemanticRole.INSTRUMENT],
                optional_roles=[SemanticRole.LOCATION, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.PATIENT: "noun",
                        SemanticRole.INSTRUMENT: "instrument",  # Use instrument category if available
                        SemanticRole.LOCATION: "location",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.1
            ),
            "temporal_action": SemanticFrame(
                frame_name="temporal_action",
                core_roles=[SemanticRole.AGENT, SemanticRole.PATIENT, SemanticRole.TIME],
                optional_roles=[SemanticRole.LOCATION],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.PATIENT: "noun",
                        SemanticRole.TIME: "temporal",
                        SemanticRole.LOCATION: "location"
                },
                weight=0.1
            ),
            "complex_motion": SemanticFrame(
                frame_name="complex_motion",
                core_roles=[SemanticRole.THEME, SemanticRole.SOURCE, SemanticRole.GOAL],
                optional_roles=[SemanticRole.AGENT, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.THEME: "noun",
                        SemanticRole.SOURCE: "location",
                        SemanticRole.GOAL: "location",
                        SemanticRole.AGENT: "noun",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.1
            ),
            "complex_communication": SemanticFrame(
                frame_name="complex_communication",
                core_roles=[SemanticRole.AGENT, SemanticRole.THEME, SemanticRole.EXPERIENCER],
                optional_roles=[SemanticRole.TIME, SemanticRole.LOCATION],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.THEME: "noun",
                        SemanticRole.EXPERIENCER: "noun",
                        SemanticRole.TIME: "temporal",
                        SemanticRole.LOCATION: "location"
                },
                weight=0.1
            ),
            "coordination": SemanticFrame(
                frame_name="coordination",
                core_roles=[SemanticRole.AGENT, SemanticRole.PATIENT],
                optional_roles=[SemanticRole.THEME, SemanticRole.LOCATION, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.PATIENT: "noun",
                        SemanticRole.THEME: "noun",  # For second coordinated object
                        SemanticRole.LOCATION: "location",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.1
            ),
            "transfer": SemanticFrame(
                frame_name="transfer",
                core_roles=[SemanticRole.AGENT, SemanticRole.THEME, SemanticRole.GOAL],
                optional_roles=[SemanticRole.SOURCE, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.THEME: "noun",
                        SemanticRole.GOAL: "noun",
                        SemanticRole.SOURCE: "noun",
                        SemanticRole.TIME: "temporal"
                },
                    weight=0.1
            ),
            "multi_participant": SemanticFrame(
                frame_name="multi_participant",
                core_roles=[SemanticRole.AGENT, SemanticRole.PATIENT, SemanticRole.EXPERIENCER],
                optional_roles=[SemanticRole.THEME, SemanticRole.LOCATION, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.PATIENT: "noun",
                        SemanticRole.EXPERIENCER: "noun",
                        SemanticRole.THEME: "noun",
                        SemanticRole.LOCATION: "location",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.1
            ),
            "experiential": SemanticFrame(
                frame_name="experiential",
                core_roles=[SemanticRole.EXPERIENCER, SemanticRole.THEME],
                optional_roles=[SemanticRole.TIME, SemanticRole.LOCATION],
                pos_mapping={
                        SemanticRole.EXPERIENCER: "noun",
                        SemanticRole.THEME: "noun",
                        SemanticRole.TIME: "temporal",
                        SemanticRole.LOCATION: "location"
                },
                    weight=0.1
            ),
            "possession": SemanticFrame(
                frame_name="possession",
                core_roles=[SemanticRole.AGENT, SemanticRole.THEME],
                optional_roles=[SemanticRole.SOURCE, SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.AGENT: "noun",
                        SemanticRole.THEME: "noun",
                        SemanticRole.SOURCE: "noun",
                        SemanticRole.TIME: "temporal"
                },
                    weight=0.1
            ),
            "location_static": SemanticFrame(
                frame_name="location_static",
                core_roles=[SemanticRole.THEME, SemanticRole.LOCATION],
                optional_roles=[SemanticRole.TIME],
                pos_mapping={
                        SemanticRole.THEME: "noun",
                        SemanticRole.LOCATION: "location",
                        SemanticRole.TIME: "temporal"
                },
                weight=0.1
            )
        }


    @staticmethod
    def get_standard_roles_count() -> int:
        """
        Returns the count of standard semantic roles.
        """
        return len(SemanticRole)
