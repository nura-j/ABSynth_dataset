import random
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import StrEnum
import json
import csv
import os
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
            random_seed: Random seed for reproducibility - should already be set in main script but just in case
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
        results = []

        # Select subset of sentences to intervene on
        if self.all:
            target_indices = list(range(len(self.corpus)))
        else:
            target_indices = random.sample(range(len(self.corpus)), self.subset_size)

        print('Applying interventions to {} sentences...'.format(len(target_indices)))
        print('Target indices:', target_indices)
        print('Intervention type:', self.intervention_type.value)
        print('Subset size:', self.subset_size)
        print('Total sentences in corpus:', len(self.corpus))

        for idx in target_indices:
            sentence_data = self.corpus[idx]

            try:
                if self.intervention_type == InterventionType.SYNONYMIC_SUBSTITUTION:
                    result = self.synonymic_substitution(idx)
                elif self.intervention_type == InterventionType.ROLE_VIOLATION:
                    result = self.role_violation(idx)
                elif self.intervention_type == InterventionType.ELIMINATION:
                    result = self.elimination(idx)
                else:
                    raise ValueError(f"Unknown intervention type: {self.intervention_type}")

                if result:
                    results.append(result)
                    self.intervention_stats['successful_interventions'] += 1

            except Exception as e:
                print(f"Failed to apply intervention to sentence {idx}: {e}")
                self.intervention_stats['failed_interventions'] += 1

            self.intervention_stats['total_interventions'] += 1
        return results

    def synonymic_substitution(self, idx: int) -> Optional[InterventionResult]:
        """
        Apply synonymic substitution intervention - replace a word with a synonym
        from the same semantic cluster while preserving the semantic role.

        Args:
            idx: Index of sentence in corpus

        Returns:
            InterventionResult or None if intervention failed
        """
        sentence_data = self.corpus[idx]
        sentence = sentence_data['sentence']
        semantic_roles = sentence_data.get('semantic_roles', {})
        # print("Applying synonymic substitution to sentence:", sentence)
        # print("Semantic roles:", semantic_roles)


        if not semantic_roles:
            return None

        available_args = list(semantic_roles.keys())
        # print("Available semantic roles:", available_args)

        if not available_args:
            return None

        target_arg = random.choice(available_args)
        role_info = semantic_roles[target_arg]

        # print("Selected target argument:", target_arg)
        # print("Role info:", role_info)

        target_word = role_info['word']
        # print("Target word for substitution:", target_word)
        target_role = role_info['role']
        # print("Target role for substitution:", target_role)
        target_position = role_info['position']
        # print("Target position in sentence:", target_position)

        # Find synonyms from the same semantic cluster
        synonym = self._find_synonym(target_word)
        print("Found synonym:", synonym)

        if not synonym or synonym == target_word:
            return None

        # Replace the word in the sentence
        words = sentence.split()
        if target_position < len(words):
            modified_words = words.copy()
            modified_words[target_position] = synonym
            modified_sentence = ' '.join(modified_words)
            print("Modified sentence:", modified_sentence)
            return InterventionResult(
                original_sentence=sentence,
                modified_sentence=modified_sentence,
                intervention_type=self.intervention_type.value,
                target_role=target_role,
                target_word=target_word,
                replacement_word=synonym,
                position=target_position,
                metadata={
                    'sentence_idx': idx,
                    'semantic_frame': sentence_data.get('metadata', {}).get('frame'),
                    'intervention_success': True
                }
            )

        return None



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

    def _find_synonym(self, target_word: str) -> Optional[str]:
        """
        Find a synonym for the target word within the same semantic cluster.

        Args:
            target_word: Word to find synonym for
        Returns:
            Synonym word or None if not found
        """
        # Extract word category from synthetic word (e.g. "noun1" -> "noun")
        word_category = self._extract_word_category(target_word)
        if word_category not in self.semantic_clusters: # we use a fallback here - rand word from the lexicon
            if word_category in self.lexicon_words:
                available_words = [w for w in self.lexicon_words[word_category] if w != target_word]
                if available_words:
                    return random.choice(available_words)
                else:
                    return None
            else:
                return None

        target_cluster = None
        for cluster_name, cluster_words in self.semantic_clusters[word_category].items():
            if target_word in cluster_words:
                target_cluster = cluster_words
                break
        if target_cluster:
            # Filter out the target word itself
            synonyms = [w for w in target_cluster if w != target_word]
            if synonyms:
                return random.choice(synonyms)
            else:
                return None
        else:
            return None


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

    @staticmethod
    def _extract_word_category(word: str) -> str:
        """
        Extract the category from a synthetic word (e.g., "noun1" -> "noun").

        Args:
            word: Synthetic word

        Returns:
            Word category
        """
        # Remove digits and common suffixes
        category = re.sub(r'\d+$', '', word)  # Remove trailing digits
        category = re.sub(r'(ed|s|ing)$', '', category)  # Remove common suffixes
        return category

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
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if format == "json":
            export_data = {
                "intervention_type": self.intervention_type.value,
                "intervention_statistics": self.get_intervention_statistics(),
                "results": [
                    {
                        "original_sentence": r.original_sentence,
                        "modified_sentence": r.modified_sentence,
                        "intervention_type": r.intervention_type,
                        "target_role": r.target_role,
                        "target_word": r.target_word,
                        "replacement_word": r.replacement_word,
                        "position": r.position,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

        elif format == "csv":
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "original_sentence", "modified_sentence", "intervention_type",
                    "target_role", "target_word", "replacement_word", "position",
                    "sentence_idx", "semantic_frame"
                ])

                for r in results:
                    writer.writerow([
                        r.original_sentence, r.modified_sentence, r.intervention_type,
                        r.target_role, r.target_word, r.replacement_word, r.position,
                        r.metadata.get('sentence_idx'), r.metadata.get('semantic_frame')
                    ])


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