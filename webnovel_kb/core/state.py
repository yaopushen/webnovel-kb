"""State persistence management."""
import json
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

import networkx as nx

from webnovel_kb.data_models import (
    NovelMeta, StyleProfile, PlotPattern,
    Entity, Relationship, WritingTemplate,
)

logger = logging.getLogger("webnovel-kb")


class StateManager:
    """状态持久化管理器。"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all(
        self,
        novels: Dict[str, NovelMeta],
        style_profiles: Dict[str, StyleProfile],
        plot_patterns: list,
        entities: Dict[str, Entity],
        relationships: list,
        writing_templates: list,
        graph: nx.DiGraph
    ) -> None:
        """加载所有状态到传入的容器中。"""
        state_file = self.data_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for k, v in state.get("novels", {}).items():
                novels[k] = NovelMeta(**v)
        
        graph_file = self.data_dir / "knowledge_graph.json"
        if graph_file.exists():
            self._load_graph(graph_file, graph)
        
        patterns_file = self.data_dir / "plot_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, "r", encoding="utf-8") as f:
                for p in json.load(f):
                    plot_patterns.append(PlotPattern(**p))
        
        styles_file = self.data_dir / "style_profiles.json"
        if styles_file.exists():
            with open(styles_file, "r", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    style_profiles[k] = StyleProfile(**v)
        
        entities_file = self.data_dir / "entities.json"
        if entities_file.exists():
            with open(entities_file, "r", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    entities[k] = Entity(**v)
        
        rels_file = self.data_dir / "relationships.json"
        if rels_file.exists():
            with open(rels_file, "r", encoding="utf-8") as f:
                for r in json.load(f):
                    relationships.append(Relationship(**r))
        
        templates_file = self.data_dir / "writing_templates.json"
        if templates_file.exists():
            with open(templates_file, "r", encoding="utf-8") as f:
                for t in json.load(f):
                    writing_templates.append(WritingTemplate(**t))
    
    def save_all(
        self,
        novels: Dict[str, NovelMeta],
        style_profiles: Dict[str, StyleProfile],
        plot_patterns: list,
        entities: Dict[str, Entity],
        relationships: list,
        writing_templates: list,
        graph: nx.DiGraph
    ) -> None:
        """保存所有状态。"""
        self._backup_state_files()
        
        state = {"novels": {k: asdict(v) for k, v in novels.items()}}
        with open(self.data_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        self._save_graph(graph)
        
        with open(self.data_dir / "plot_patterns.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in plot_patterns], f, ensure_ascii=False, indent=2)
        
        with open(self.data_dir / "style_profiles.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in style_profiles.items()}, f, ensure_ascii=False, indent=2)
        
        with open(self.data_dir / "entities.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in entities.items()}, f, ensure_ascii=False, indent=2)
        
        with open(self.data_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in relationships], f, ensure_ascii=False, indent=2)
        
        with open(self.data_dir / "writing_templates.json", "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in writing_templates], f, ensure_ascii=False, indent=2)
    
    def _backup_state_files(self) -> None:
        """备份现有状态文件。"""
        backup_dir = self.data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for fname in ["state.json", "knowledge_graph.json", "plot_patterns.json",
                      "style_profiles.json", "entities.json", "relationships.json",
                      "writing_templates.json"]:
            src = self.data_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / f"{fname}.{timestamp}.bak")
    
    def _save_graph(self, graph: nx.DiGraph) -> None:
        data = nx.node_link_data(graph)
        with open(self.data_dir / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_graph(self, path: Path, graph: nx.DiGraph) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = nx.node_link_graph(data, directed=True)
        graph.clear()
        graph.add_nodes_from(loaded.nodes(data=True))
        graph.add_edges_from(loaded.edges(data=True))
