# WebNovel Knowledge Base - 开发路线图

## 当前版本：v1.0

### 核心功能
- [x] 小说导入与分块（支持章节识别）
- [x] 语义检索（BGE-small-zh + FAISS）
- [x] BM25 关键词检索（Tantivy）
- [x] 混合搜索（向量 + BM25 融合）
- [x] LLM Rerank 精排
- [x] 输出降噪（compact/clean 模式）
- [x] 去重机制（子串检测）
- [x] 实体提取与管理
- [x] 关系图谱构建
- [x] 情节模式自动提取
- [x] 写作模板提取
- [x] 风格分析（节奏、对话比、钩子密度）
- [x] MCP SSE/stdio 双模式支持

---

## v1.1 计划

### 功能增强
- [ ] 批量导入优化（进度条、断点续传）
- [ ] 导出功能（JSON/Markdown）
- [ ] 搜索历史记录
- [ ] 相似小说推荐

### 性能优化
- [ ] 索引增量更新（无需全量重建）
- [ ] 查询缓存层
- [ ] 内存使用优化

---

## v2.0 计划 - 架构重构

### 代码重构

当前 `server.py` 已达 2200+ 行，需要进行模块拆分：

```
webnovel_kb/
├── core/
│   ├── knowledge_base.py    # 协调层
│   ├── indexer.py           # 索引构建
│   ├── chunker.py           # 文本分块
│   └── state.py             # 状态持久化
├── search/
│   ├── semantic.py
│   ├── bm25.py
│   ├── hybrid.py
│   └── unified.py
├── extraction/
│   ├── entities.py
│   ├── plot_patterns.py
│   └── writing_templates.py
├── analysis/
│   └── style.py
└── api/
    └── mcp_tools.py
```

### 单元测试
- [ ] 核心模块测试覆盖率 > 80%
- [ ] 集成测试

### 插件系统
- [ ] 支持自定义提取器
- [ ] 支持自定义分析器

---

## v3.0 愿景

### 多模型支持
- [ ] OpenAI Embedding 适配
- [ ] 本地模型（Ollama）支持
- [ ] 多嵌入模型共存

### 知识增强
- [ ] 网文套路百科（内置常见模式库）
- [ ] 热梗检测与标注
- [ ] 灵感推荐系统

### 协作功能
- [ ] 多用户支持
- [ ] 素材共享
- [ ] 版本管理

---

## 技术债务

| 问题 | 优先级 | 计划版本 |
|---|---|---|
| server.py 过大 | 高 | v2.0 |
| 缺少单元测试 | 高 | v2.0 |
| 异步任务无持久化 | 中 | v1.1 |
| 错误处理不统一 | 中 | v1.1 |
| 日志过于冗长 | 低 | v1.1 |

---

## 贡献指南

欢迎贡献代码！请遵循：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范
- Python 3.10+
- 类型注解
- 单文件不超过 500 行
- 函数不超过 50 行

---

## 更新日志

### v1.0.0 (2025-04-30)
- 首个正式发布版本
- 12 个 MCP 工具
- 降噪输出支持
- SSE + stdio 双模式
