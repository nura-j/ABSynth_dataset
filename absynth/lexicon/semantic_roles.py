from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class SemanticRole(Enum):
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
                }
            ),

            "intransitive_action": SemanticFrame(
                frame_name="intransitive_action",
                core_roles=[SemanticRole.AGENT],
                optional_roles=[SemanticRole.LOCATION, SemanticRole.TIME],
                pos_mapping={
                    SemanticRole.AGENT: "noun",
                    SemanticRole.LOCATION: "location",
                    SemanticRole.TIME: "temporal"
                }
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
                }
            ),

            "motion": SemanticFrame(
                frame_name="motion",
                core_roles=[SemanticRole.THEME],
                optional_roles=[SemanticRole.SOURCE, SemanticRole.GOAL, SemanticRole.TIME],
                pos_mapping={
                    SemanticRole.THEME: "noun",
                    SemanticRole.SOURCE: "location",
                    SemanticRole.GOAL: "location",
                    SemanticRole.TIME: "temporal"
                }
            )
        }