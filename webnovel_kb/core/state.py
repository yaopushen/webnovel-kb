"""State persistence for knowledge base data."""
import pickle
from pathlib import Path
from typing import Dict, List

import networkx as nx
from webnovel_kb.data_models import (
    NovelMeta, StyleProfile, PlotPattern,
    Entity, Relationship, WritingTemplate
)
from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("core.state")


class StateManager:
    """状态持久化管理器。"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state_dir = data_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self, novels: Dict[str, NovelMeta], style_profiles: Dict[str, StyleProfile],
                 plot_patterns: List[PlotPattern], entities: Dict[str, Entity],
                 relationships: List[Relationship], writing_templates: List[WritingTemplate],
                 graph: nx.DiGraph) -> None:
        """保存所有状态。"""
        self._save_pickle("novels.pkl", novels)
        self._save_pickle("style_profiles.pkl", style_profiles)
        self._save_pickle("plot_patterns.pkl", plot_patterns)
        self._save_pickle("entities.pkl", entities)
        self._save_pickle("relationships.pkl", relationships)
        self._save_pickle("writing_templates.pkl", writing_templates)
        self._save_pickle("graph.pkl", graph)

    def load_all(self, novels: Dict[str, NovelMeta], style_profiles: Dict[str, StyleProfile],
                 plot_patterns: List[PlotPattern], entities: Dict[str, Entity],
                 relationships: List[Relationship], writing_templates: List[WritingTemplate],
                 graph: nx.DiGraph) -> None:
        """加载所有状态。"""
        self._load_pickle("novels.pkl", novels)
        self._load_pickle("style_profiles.pkl", style_profiles)
        self._load_pickle("plot_patterns.pkl", plot_patterns)
        self._load_pickle("entities.pkl", entities)
        self._load_pickle("relationships.pkl", relationships)
        self._load_pickle("writing_templates.pkl", writing_templates)
        self._load_pickle("graph.pkl", graph, nx.DiGraph())

    def _save_pickle(self, filename: str, data) -> None:
        """保存 pickle 文件。"""
        path = self.state_dir / filename
        try:
            with open(path.with_suffix('.pkl.tmp'), "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            path.with_suffix('.pkl.tmp').replace(path)
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")

    def _load_pickle(self, filename: str, target, default=None) -> None:
        """加载 pickle 文件到目标容器。"""
        path = self.state_dir / filename
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(target, dict):
                target.clear()
                target.update(data)
            elif isinstance(target, list):
                target.clear()
                target.extend(data)
            else:
                pass
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            if default is not None:
                if isinstance(target, dict):
                    target.clear()
                    target.update(default)
                elif isinstance(target, list):
                    target.clear()
                    target.extend(default)
