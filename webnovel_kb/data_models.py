"""Data models for web novel knowledge base."""
from dataclasses import dataclass, field


@dataclass
class NovelMeta:
    title: str
    author: str
    genre: str
    word_count: int = 0
    file_path: str = ""
    chunk_count: int = 0
    chapter_count: int = 0
    first_chapter: int = 0
    last_chapter: int = 0


@dataclass
class StyleProfile:
    avg_sentence_len: float = 0.0
    dialogue_ratio: float = 0.0
    inner_monologue_ratio: float = 0.0
    description_ratio: float = 0.0
    action_ratio: float = 0.0
    narrative_perspective: str = ""

    section_breakdown: list = field(default_factory=list)
    humor_scenes: list = field(default_factory=list)
    sample_passages: list = field(default_factory=list)

    ai_fingerprint_score: float = 0.0
    oral_score: float = 0.0
    chapter_hook_rate: float = 0.0
    pace_type: str = ""

    humor_markers: list = field(default_factory=list)
    pacing_info: dict = field(default_factory=dict)
    humor_type: str = ""
    tension_relax_pattern: str = ""


@dataclass
class PlotPattern:
    pattern_type: str
    description: str
    source_novel: str
    source_chapter: str
    before_context: str = ""
    pattern_text: str = ""
    after_context: str = ""
    effectiveness: str = ""


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str
    source_novel: str
    role: str = ""
    first_appearance: str = ""
    arc: str = ""
    attributes: dict = field(default_factory=dict)


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: str
    description: str
    source_novel: str
    evolution: str = ""


@dataclass
class WritingTemplate:
    template_type: str
    scene_type: str
    structure: str
    key_beats: list
    source_novel: str
    source_chapter: str = ""
    example_text: str = ""
    effectiveness: str = ""


@dataclass
class ChapterOutline:
    outline_id: str
    novel_title: str
    chapter: int
    outline_type: str
    content: str
    created_at: str = ""
    updated_at: str = ""
    tags: list = field(default_factory=list)


@dataclass
class KnowledgeEntry:
    entry_id: str
    title: str
    content: str
    category: str = "research"  # market_analysis/style_guide/writing_technique/user_experience/research/query_insight
    tags: list = field(default_factory=list)
    source: str = ""
    created_at: str = ""
    access_count: int = 0
    auto_generated: bool = False
    archived: bool = False
