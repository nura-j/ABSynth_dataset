
# sentence/__init__.py
"""
Sentence package for semantic frame-based sentence generation.
Provides template management and sentence generation with linguistic annotations.
"""
from .frame_manager import TemplateManager
from .sentence_generator import SentenceGenerator
from .linguistic_annotator import LinguisticAnnotator

__all__ = ['TemplateManager', 'SentenceGenerator', 'LinguisticAnnotator']
