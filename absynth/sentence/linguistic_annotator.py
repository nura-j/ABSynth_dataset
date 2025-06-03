from typing import Dict, List, Any, Optional
from dataclasses import dataclass



@dataclass
class LinguisticAnnotation:
    """Container for comprehensive linguistic annotations."""
    pos_tags: List[str]
    dependency_parse: List[Dict[str, Any]]
    constituency_parse: str
    semantic_roles: Dict[str, str] # todo via https://stanfordnlp.github.io/stanza/available_models.html
    formal_semantics: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        Used for saving annotations in a structured format.
        """
        return {
            "pos_tags": self.pos_tags,
            # "dependency_parse": self.dependency_parse,
            # "constituency_parse": self.constituency_parse,
            "semantic_roles": self.semantic_roles,
            "formal_semantics": self.formal_semantics
        }


class LinguisticAnnotator:
    """
    Provides comprehensive linguistic annotations for generated sentences.
    """

    def __init__(self):
        """Initialize the linguistic annotator."""
        self.pos_tag_mapping = self._create_pos_mapping()

    def _create_pos_mapping(self) -> Dict[str, str]:
        """Create mapping from word categories to POS tags.
        We follow the Penn Treebank POS tag set for consistency: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """

        return {
            "noun": "NN",
            "transitive_verb": "VB",
            "intransitive_verb": "VB",
            "communication_verb": "VB",
            "motion_verb": "VB",
            "change_verb": "VB",
            "adjective": "JJ",
            "adverb": "RB",
            "location": "NN",
            "temporal": "NN",
            "preposition": "IN",
            "conjunction": "CC",
            "determiner": "DT",
            "cardinal_number": "CD",
            "numeral": "CD",
        }

    def annotate_sentence(self, sentence: str, words: List[str],
                          role_assignments: Dict, frame_name: str) -> LinguisticAnnotation:
        """
        Generate comprehensive linguistic annotations for a sentence.

        Args:
            sentence: The generated sentence
            words: List of words in the sentence
            role_assignments: Semantic role assignments
            frame_name: Semantic frame name

        Returns:
            LinguisticAnnotation object with all annotations
        """
        # Generate POS tags
        pos_tags = self._generate_pos_tags(words)

        # # Generate dependency parse
        # dependency_parse = self._generate_dependency_parse(words, pos_tags, role_assignments)
        #
        # # Generate constituency parse
        # constituency_parse = self._generate_constituency_parse(words, pos_tags, role_assignments)

        # Extract semantic roles
        semantic_roles = {info["word"]: info["role"] for info in role_assignments.values()}

        # Create formal semantics
        formal_semantics = self._generate_formal_semantics(role_assignments, frame_name)

        return LinguisticAnnotation(
            pos_tags=pos_tags,
            semantic_roles=semantic_roles,
            formal_semantics=formal_semantics
        )

    def _generate_pos_tags(self, words: List[str]) -> List[str]:
        """Generate POS tags for words."""
        pos_tags = []
        for word in words:
            # Extract category from synthetic word (e.g., "noun1" -> "noun")
            category = ''.join([c for c in word if not c.isdigit()])
            if category.endswith('ed') or category.endswith('s'):
                # Handle verb inflections
                base_category = category.rstrip('eds')
                if base_category in self.pos_tag_mapping:
                    if category.endswith('ed'):
                        pos_tags.append("VBD")  # Past tense
                    elif category.endswith('ing'):
                        pos_tags.append("VBG")
                    else:
                        pos_tags.append(self.pos_tag_mapping.get(base_category, "NN"))
                else:
                    pos_tags.append("NN")
            else:
                pos_tags.append(self.pos_tag_mapping.get(category, "NN"))

        return pos_tags


    def _generate_formal_semantics(self, role_assignments: Dict, frame_name: str) -> str:
        """Generate formal semantic representation using lambda calculus."""
        variables = []
        predicates = []

        for arg, info in role_assignments.items():
            var = f"x{info['position']}"
            variables.append(var)
            predicates.append(f"{info['role']}({var})")

        if variables and predicates:
            lambda_vars = " ".join(variables)
            predicate_conj = " ∧ ".join(predicates)
            return f"λ{lambda_vars}.∃e.{frame_name}(e) ∧ {predicate_conj}"
        else:
            return f"λe.{frame_name}(e)"