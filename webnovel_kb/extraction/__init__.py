"""Extraction modules for webnovel_kb."""
from .entities import EntityExtractor
from .plot_patterns import PlotPatternExtractor
from .writing_templates import WritingTemplateExtractor
from .scene_patterns import ScenePatternExtractor

__all__ = ["EntityExtractor", "PlotPatternExtractor", "WritingTemplateExtractor", "ScenePatternExtractor"]
