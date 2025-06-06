# corpus/__init__.py
"""
Corpus package for synthetic corpus generation and evaluation.
Provides corpus generation, evaluation, and analysis tools.
"""

from .synthetic_corpus_generator import SyntheticCorpusGenerator
from .corpus_evaluator import CorpusEvaluator

__all__ = ['SyntheticCorpusGenerator', 'CorpusEvaluator']
