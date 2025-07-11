import copy
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import StrEnum
import json
import csv
import os
from absynth import SynthCorpus
from absynth.lexicon.semantic_roles import SemanticRole


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
    original_metadata: Optional[Dict[str, Any]] = None
    modified_metadata: Optional[Dict[str, Any]] = None


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

        # print('Applying interventions to {} sentences...'.format(len(target_indices)))
        # print('Target indices:', target_indices)
        # print('Intervention type:', self.intervention_type.value)
        # print('Subset size:', self.subset_size)
        # print('Total sentences in corpus:', len(self.corpus))

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
        sentence_data_modified = copy.deepcopy(sentence_data)
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
        # print("Found synonym:", synonym)

        if not synonym or synonym == target_word:
            return None

        # Replace the word in the sentence
        words = sentence.split()
        if target_position < len(words):
            modified_words = words.copy()
            modified_words[target_position] = synonym
            modified_sentence = ' '.join(modified_words)
            sentence_data_modified['sentence'] = modified_sentence

            if target_arg in sentence_data_modified.get('semantic_roles', {}):
                sentence_data_modified['semantic_roles'][target_arg]['word'] = synonym

            annotations = sentence_data_modified.get('linguistic_annotations', {})
            if 'semantic_roles' in annotations:
                del annotations['semantic_roles'][target_word]
                annotations['semantic_roles'][synonym] = target_role

            if 'formal_semantics' in annotations:
                annotations['formal_semantics'] = annotations['formal_semantics'].replace(target_word, synonym)
            if 'semantics' in sentence_data_modified:
                # Remove the word from the semantics representation
                sentence_data_modified['semantics'] = sentence_data_modified['semantics'].replace(target_word, synonym)

            # print('Original sentence:', sentence)
            # for key, value in sentence_data.items():
            #     print(f"{key}: {value}")
            # print("Modified sentence:", modified_sentence)
            # for key, value in sentence_data_modified.items():
            #     print(f"Metadata {key}: {value}")
            # print("Modified sentence:", modified_sentence)
            return InterventionResult(
                original_sentence=sentence,
                modified_sentence=modified_sentence,
                original_metadata=sentence_data,
                modified_metadata=sentence_data_modified,
                intervention_type=self.intervention_type.value,
                target_role=target_role,
                target_word=target_word,
                replacement_word=synonym,
                position=target_position,
                metadata={
                    'sentence_idx': idx,
                    'semantic_frame': sentence_data.get('metadata', {}).get('frame'),
                    'intervention_success': True,
                    'substitution_type': 'synonymic'
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
        sentence_data = self.corpus[idx]
        sentence_data_modified = copy.deepcopy(sentence_data)

        sentence = sentence_data['sentence']
        semantic_roles = sentence_data.get('semantic_roles', {})

        if not semantic_roles:
            return None

        # Select a random argument position to violate
        available_args = list(semantic_roles.keys())
        if not available_args:
            return None

        target_arg = random.choice(available_args)
        role_info = semantic_roles[target_arg]

        target_word = role_info['word']
        target_role = role_info['role']
        target_position = role_info['position']
        # print("Applying role violation to sentence:", sentence)

        # print("Target argument for violation:", target_arg)
        # print("Role info:", role_info)

        # Find a word that violates the expected role
        violating_word = self._find_role_violating_word(target_role)

        if not violating_word:
            return None

        # Replace the word with the violating word
        words = sentence.split()
        # print("Target word for violation:", target_word)
        # print('original words:', words)
        if target_position < len(words):
            modified_words = words.copy()
            modified_words[target_position] = violating_word
            # print("Modified words after violation:", modified_words)
            modified_sentence = ' '.join(modified_words)
            sentence_data_modified['sentence'] = modified_sentence

            # Update semantic_roles
            sentence_data_modified['semantic_roles'][target_arg]['word'] = violating_word
            annotations = sentence_data_modified.get('linguistic_annotations', {})
            if 'semantic_roles' in annotations:
                annotations['semantic_roles'].pop(target_word, None)
                annotations['semantic_roles'][violating_word] = target_role

            if 'formal_semantics' in annotations:
                annotations['formal_semantics'] = annotations['formal_semantics'].replace(target_word, violating_word)

            return InterventionResult(
                original_sentence=sentence,
                modified_sentence=modified_sentence,
                original_metadata=sentence_data,
                modified_metadata=sentence_data_modified,
                intervention_type=self.intervention_type.value,
                target_role=target_role,
                target_word=target_word,
                replacement_word=violating_word,
                position=target_position,
                metadata={
                    'sentence_idx': idx,
                    'semantic_frame': sentence_data.get('metadata', {}).get('frame'),
                    'violation_type': 'role_mismatch',
                    'intervention_success': True
                }
            )

        return None


    def elimination(self, idx: int) -> Optional[InterventionResult]:
        """
        Apply elimination intervention - systematically remove argument slots
        to test how models handle incomplete argument structures.

        Args:
            idx: Index of sentence in corpus

        Returns:
            InterventionResult or None if intervention failed
        """
        sentence_data = self.corpus[idx]
        sentence_data_modified = copy.deepcopy(sentence_data)

        sentence = sentence_data['sentence']
        semantic_roles = sentence_data.get('semantic_roles', {})
        # print('Original sentence1:')
        # for key, value in sentence_data.items():
        #     print(f"{key}: {value}")
        # print("Applying elimination intervention to sentence:", sentence,)
        # print("Semantic roles:", semantic_roles)
        # print('sentence data:', sentence_data)
        # for key, value in sentence_data.items():
        #     print(f"{key}: {value}")

        if not semantic_roles:
            return None

        # Select a random argument to eliminate
        available_args = list(semantic_roles.keys())
        if not available_args:
            return None

        target_arg = random.choice(available_args)
        role_info = semantic_roles[target_arg]
        target_word = role_info['word']
        target_role = role_info['role']
        target_position = role_info['position']

        # Remove the word and handle adjacent function words
        words = sentence.split()
        if target_position >= len(words):
            return None

        modified_words = words.copy()
        # Remove the target word
        del modified_words[target_position]
        # updating the sentence data to reflect the remova

        # print('Original sentence:', sentence)
        # print("Modified words after removal:", modified_words)
        # Clean up the sentence by updating the others
        modified_words, removed_indices = self._clean_up_words(modified_words, target_position)
        removed_indices = sorted(set(removed_indices))
        # print("Removed indices after cleanup:", removed_indices)
        # print("Modified words after cleanup:", modified_words)
        # print('*' * 20)
        """
        sentence: noun12 motion_verb14s location60
        semantic_roles: {'arg0': {'word': 'noun12', 'role': 'Theme', 'position': 0}, 'arg1': {'word': 'location60', 'role': 'Goal', 'position': 2}}
        semantics: ∃e.motion(e) ∧ Theme(e, noun12) ∧ Goal(e, location60)
        linguistic_annotations: {'pos_tags': ['NN', 'VB', 'NN'], 
                                'semantic_roles': {'noun12': 'Theme', 'location60': 'Goal'}, 
                                'formal_semantics': 'λx0 x2.∃e.motion(e) ∧ Theme(x0) ∧ Goal(x2)'}
        metadata: {'complexity': 'simple', 
                'frame': 'motion', 
                'template': {'frame': 'motion', 'args': ['arg0', 'verb', 'arg1'], 
                            'roles': {'arg0': <SemanticRole.THEME: 'Theme'>, 'arg1': <SemanticRole.GOAL: 'Goal'>}, 
                            'weight': 0.16666666666666666}, 
                            'length': 3, 
                            'entropy_profile': [6.949784284662096, 6.949784284662096, 6.949784284662096], 
                            'avg_entropy': 6.949784284662097}
        corpus_metadata: {'target_frame': 'basic_intransitive', 'sentence_id': 1}
        """
        sentence_data_modified['sentence'] = ' '.join(modified_words)

        #################################### # Update semantic roles ####################################
        for arg_key, role in list(sentence_data_modified.get('semantic_roles', {}).items()):
            print('Processing semantic role:', arg_key, 'with role info:', role)
            if role['position'] in removed_indices:
                # Remove the semantic role entry if its position was removed
                # print(f"Removing semantic role {arg_key} at position {role['position']}")
                removed_word = role['word']
                sentence_data_modified['semantic_roles'].pop(arg_key, None)
                annotations = sentence_data_modified.get('linguistic_annotations', {})
                if 'pos_tags' in annotations and role['position'] < len(annotations['pos_tags']):
                    del annotations['pos_tags'][role['position']]

                if 'semantic_roles' in annotations and removed_word in annotations['semantic_roles']:
                    del annotations['semantic_roles'][removed_word]

                if 'formal_semantics' in annotations:
                    annotations['formal_semantics'] = annotations['formal_semantics'].replace(removed_word,
                                                                                              'DELETED_WORD')
                if 'semantics' in sentence_data_modified:
                    # Remove the word from the semantics representation
                    sentence_data_modified['semantics'] = sentence_data_modified['semantics'].replace(removed_word,
                                                                                                      'DELETED_WORD')
                # Update metadata
                metadata = sentence_data_modified.get('metadata', {})
                if 'length' in metadata:
                    metadata['length'] = max(0, metadata['length'] - 1)
                if 'entropy_profile' in metadata:
                    if role['position'] < len(metadata['entropy_profile']):
                        del metadata['entropy_profile'][role['position']]
                    if metadata['entropy_profile']:
                        metadata['avg_entropy'] = sum(metadata['entropy_profile']) / len( metadata['entropy_profile'])
                    else:
                        metadata['avg_entropy'] = 0.0 #arg_key
                if 'roles' in metadata:
                    if arg_key in metadata['roles']:
                        del metadata['roles'][arg_key]
                if 'args' in metadata:
                    if arg_key in metadata['args']:
                        metadata['args'].remove(arg_key)

        if len(modified_words) == 0:
            return None

        modified_sentence = ' '.join(modified_words)
        words_removed = len(words) - len(modified_words)
        # print('Original sentence2:')
        # for key, value in sentence_data.items():
        #     print(f"{key}: {value}")
        # print('Updated data after removal:')
        # for key, value in sentence_data_modified.items():
        #     print(f"{key}: {value}")
        return InterventionResult(
            original_sentence=sentence,
            modified_sentence=modified_sentence,
            original_metadata=sentence_data,
            modified_metadata=sentence_data_modified,
            intervention_type=self.intervention_type.value,
            target_role=target_role,
            target_word=target_word,
            replacement_word=None,
            position=target_position,
            metadata={
                'sentence_idx': idx,
                'semantic_frame': sentence_data.get('metadata', {}).get('frame'),
                'elimination_type': 'argument_removal',
                'words_removed': words_removed,
                'intervention_success': True
            }
        )



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
        '''
            AGENT = "Agent"  # The doer of the action
            PATIENT = "Patient"  # The entity affected by the action
            THEME = "Theme"  # The entity moved or described
            EXPERIENCER = "Experiencer"  # The entity experiencing something
            INSTRUMENT = "Instrument"  # The tool used
            LOCATION = "Location"  # Where something happens
            SOURCE = "Source"  # Starting point
            GOAL = "Goal"  # End point
            TIME = "Time"  # When something happens
        '''
        # raise NotImplementedError("Subclasses must implement _find_role_violating_word method")
        role_violations = {
            SemanticRole.AGENT.value: ['location', 'temporal', 'instrument'],
            SemanticRole.PATIENT.value: ['temporal', 'adverb', 'instrument'],
            SemanticRole.THEME.value: ['adverb', 'adjective', 'temporal'],
            SemanticRole.LOCATION.value: ['instrument', 'noun', 'temporal'],
            SemanticRole.TIME.value: ['location', 'instrument', 'noun'],
            SemanticRole.INSTRUMENT.value: ['temporal', 'location', 'adverb'],
            SemanticRole.EXPERIENCER.value: ['location', 'temporal', 'instrument'],
            SemanticRole.SOURCE.value: ['temporal', 'instrument', 'adverb'],
            SemanticRole.GOAL.value: ['temporal', 'instrument', 'adverb']
        }

        violating_categories = role_violations.get(target_role, ['adverb'])

        # Select a random violating category and word
        for category in violating_categories:
            if category in self.lexicon_words and self.lexicon_words[category]:
                return random.choice(self.lexicon_words[category])

        # Fallback: use any random word from a different category
        all_categories = list(self.lexicon_words.keys())
        if all_categories:
            fallback_category = random.choice(all_categories)
            if self.lexicon_words[fallback_category]:
                return random.choice(self.lexicon_words[fallback_category])

        return None

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

    def _clean_up_words(self, words: List[str], removed_position: int) -> Tuple[List[str], List[int]]:
        """
        Clean up prepositions, conjunctions after word removal.
        "preposition" (e.g., in, on, at)
        "conjunction" (e.g., and, but, or)
        "determiner" (e.g., the, a, some)

        Args:
            words: List of words after removal
            removed_position: Position where word was removed

        Returns:
            Cleaned list of words
        """
        removed_indices = [removed_position]
        if not words:
            return words, removed_indices

        function_words = ['preposition', 'conjunction', 'determiner'] # adding more than the basic ones just in case
        def is_function_word(word: str) -> bool:
            # print("Checking if word is a function word:", word)
            # print("Extracted category:", self._extract_word_category(word))
            return self._extract_word_category(word) in function_words


        # Check position before removal point
        if removed_position > 0 and removed_position < len(words):
            # print('Removing adjacent function words at position:', removed_position)
            # print('Words before removal:', words)
            # print('len of words:', len(words))
            prev_word = words[removed_position - 1]
            # print("Previous word:", prev_word)
            if prev_word and is_function_word(prev_word):
                del words[removed_position - 1]
                removed_indices.append(removed_position - 1)
                # print('updated words after removing previous word:', words)
                removed_position -= 1

        if removed_position < len(words):
            next_word = words[removed_position]
            # print("Next word:", next_word)
            if next_word and is_function_word(next_word):
                del words[removed_position]
                removed_indices.append(removed_position+1)
                # print('updated words after removing next word:', words)

        return words, removed_indices



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
        todo: unify this with the corpus export functionality

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