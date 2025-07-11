import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import StrEnum

from absynth import SynthCorpus


class InterventionType(StrEnum):
    SYNONYMIC_SUBSTITUTION = "synonymic_substitution"
    ROLE_VIOLATION = "role_violation"
    ELIMINATION = "elimination"


@dataclass
class InterventionResult:
    """Container for intervention results with metadata."""
    original_sentence: str
    modified_sentence: str
    intervention_type: str
    target_role: Optional[str]
    target_word: Optional[str]
    replacement_word: Optional[str]
    position: Optional[int]
    metadata: Dict[str, Any]


class Intervention:
    def __init__(self, corpus: SynthCorpus, intervention_type: str,
                 all: bool = False, subset_percentage: float = 0.2,
                 random_seed: Optional[int] = None):
        """
        Base class for interventions on generated sentences.

        Args:
            corpus: Corpus object containing sentences to apply interventions on
            intervention_type: Type of intervention to apply
            all: Whether to apply intervention to all sentences or subset
            subset_percentage: Fraction of sentences to intervene on if not all
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        self.corpus = corpus
        self.intervention_type = InterventionType(intervention_type)
        self.all = all
        self.subset_percentage = subset_percentage
        self.subset_size = int(len(corpus.sentences) * subset_percentage) if not all else len(corpus.sentences)

        # Extract lexicon info for interventions
        self.lexicon_info = corpus.metadata.get('lexicon_info', {})
        self.semantic_clusters = self.lexicon_info.get('semantic_clusters', {})
        self.lexicon_words = self.lexicon_info.get('lexicon', {})

        # Track intervention statistics
        self.intervention_stats = {
            'total_interventions': 0,
            'successful_interventions': 0,
            'failed_interventions': 0,
            'intervention_by_role': {},
            'intervention_by_frame': {}
        }

    def apply_interventions(self) -> List[InterventionResult]:
        """
        Apply interventions to the corpus and return results.

        Returns:
            List of InterventionResult objects
        """
        raise NotImplementedError("Subclasses must implement apply_interventions method")

    def synonymic_substitution(self, idx: int) -> Optional[InterventionResult]:
        """
        Apply synonymic substitution intervention - replace a word with a synonym
        from the same semantic cluster while preserving the semantic role.

        Args:
            idx: Index of sentence in corpus

        Returns:
            InterventionResult or None if intervention failed
        """
        raise NotImplementedError("Subclasses must implement synonymic_substitution method")

    def role_violation(self, idx: int) -> Optional[InterventionResult]:
        """
        Apply role violation intervention - insert a role-inconsistent token
        into an argument position to test semantic role understanding.

        Args:
            idx: Index of sentence in corpus

        Returns:
            InterventionResult or None if intervention failed
        """
        raise NotImplementedError("Subclasses must implement role_violation method")

    def elimination(self, idx: int) -> Optional[InterventionResult]:
        """
        Apply elimination intervention - systematically remove argument slots
        to test how models handle incomplete argument structures.

        Args:
            idx: Index of sentence in corpus

        Returns:
            InterventionResult or None if intervention failed
        """
        raise NotImplementedError("Subclasses must implement elimination method")

    def _find_synonym(self, target_word: str, target_role: str) -> Optional[str]:
        """
        Find a synonym for the target word within the same semantic cluster.

        Args:
            target_word: Word to find synonym for
            target_role: Semantic role of the target word

        Returns:
            Synonym word or None if not found
        """
        # Extract word category from synthetic word (e.g., "noun1" -> "noun")
        raise NotImplementedError("Subclasses must implement _find_synonym method")

    def _find_role_violating_word(self, target_role: str) -> Optional[str]:
        """
        Find a word that semantically violates the expected role.

        Args:
            target_role: The semantic role being violated

        Returns:
            Word that violates the role or None
        """
        # Define role violation mappings
        raise NotImplementedError("Subclasses must implement _find_role_violating_word method")

    def _extract_word_category(self, word: str) -> str:
        """
        Extract the category from a synthetic word (e.g., "noun1" -> "noun").

        Args:
            word: Synthetic word

        Returns:
            Word category
        """
        # Remove digits and common suffixes
        raise NotImplementedError("Subclasses must implement _extract_word_category method")

    def _clean_up_function_words(self, words: List[str], removed_position: int) -> List[str]:
        """
        Clean up orphaned prepositions, conjunctions after word removal.

        Args:
            words: List of words after removal
            removed_position: Position where word was removed

        Returns:
            Cleaned list of words
        """
        raise NotImplementedError("Subclasses must implement _clean_up_function_words method")

    def generate_prompts(self, results: List[InterventionResult],
                         prompt_template: str = None) -> List[str]:
        """
        Generate prompts for mechanistic interpretability experiments.

        Args:
            results: List of intervention results
            prompt_template: Template for prompt generation

        Returns:
            List of formatted prompts
        """
        raise NotImplementedError("Subclasses must implement generate_prompts method")

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about applied interventions.

        Returns:
            Dictionary with intervention statistics
        """
        return self.intervention_stats.copy()

    def export_intervention_dataset(self, results: List[InterventionResult],
                                    output_path: str, format: str = "json"):
        """
        Export intervention results as a dataset for analysis.

        Args:
            results: List of intervention results
            output_path: Path to save the dataset
            format: Export format ("json" or "csv")
        """
        raise NotImplementedError("Subclasses must implement export_intervention_dataset method")


# Convenience functions for each intervention type
def apply_synonymic_substitution(corpus: SynthCorpus, subset_percentage: float = 0.2,
                                 random_seed: Optional[int] = None) -> List[InterventionResult]:
    """Apply synonymic substitution intervention to corpus."""
    intervention = Intervention(corpus, InterventionType.SYNONYMIC_SUBSTITUTION.value,
                                subset_percentage=subset_percentage, random_seed=random_seed)
    return intervention.apply_interventions()


def apply_role_violation(corpus: SynthCorpus, subset_percentage: float = 0.2,
                         random_seed: Optional[int] = None) -> List[InterventionResult]:
    """Apply role violation intervention to corpus."""
    intervention = Intervention(corpus, InterventionType.ROLE_VIOLATION.value,
                                subset_percentage=subset_percentage, random_seed=random_seed)
    return intervention.apply_interventions()


def apply_elimination(corpus: SynthCorpus, subset_percentage: float = 0.2,
                      random_seed: Optional[int] = None) -> List[InterventionResult]:
    """Apply elimination intervention to corpus."""
    intervention = Intervention(corpus, InterventionType.ELIMINATION.value,
                                subset_percentage=subset_percentage, random_seed=random_seed)
    return intervention.apply_interventions()