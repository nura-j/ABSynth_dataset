from dataclasses import dataclass
from typing import Dict


@dataclass
class Vocabulary:
    """
    Represents a vocabulary with a set of words and their corresponding IDs.
    """
    words: Dict[str, int] = None

    def __post_init__(self):

        if self.words is None:
            self.words = { # todo: change to thematic categories
                "noun": 200,
                "transitive_verb": 30,
                "intransitive_verb": 30,
                "communication_verb": 15,
                "motion_verb": 15,
                "change_verb": 15,
                "adjective": 25,
                "adverb": 15,
                "location": 100,
                "temporal": 20,
            }

        if not isinstance(self.words, dict):
            raise TypeError("Words must be a dictionary mapping categories to sizes.")
        if not all(isinstance(cat, str) for cat in self.words.keys()):
            raise TypeError("All keys must be strings.")
        if not all(isinstance(size, int) for size in self.words.values()):
            raise TypeError("All values must be integers.")

    def __getitem__(self, category: str) -> int:
        """
        Get the size of the vocabulary for a specific category.

        Args:
            category (str): The category of the vocabulary.

        Returns:
            int: The size of the vocabulary for the specified category.
        """
        return self.words.get(category, 0)

    def __setitem__(self, category: str, size: int)-> None:
        """
        Set the size of the vocabulary for a specific category.

        Args:
            category (str): The category of the vocabulary.
            size (int): The size to set for the specified category.
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("Size must be a non-negative integer.")
        self.words[category] = size

    def items(self) -> Dict[str, int].items:
        """Return items for compatibility."""
        return self.words.items()

    def __contains__(self, key) -> bool:
        """Allow 'in' operator."""
        return key in self.words

