# Changelog

## [1.5] - 2026-04-30

### Added
- 新增统一搜索接口，支持semantic、bm25、hybrid、rerank四种模式
- 新增实体语义搜索功能
- 新增知识库搜索功能（情节模式、写作模板）
- 新增异步提取任务支持
- 新增MCP工具集，包含12个工具
- 新增Tantivy+FAISS高性能索引支持
- 新增场景模式提取
- 新增去重和输出格式化

### Changed
- 重构为模块化架构，分离核心逻辑、搜索、提取、分析等模块
- 优化内存使用，嵌入缓存从Python list改为numpy.float32，内存占用减少约80%
- 修复语义搜索维度不匹配问题（ChromaDB默认嵌入函数维度与API模型维度不一致）
- 修复API客户端URL路径缺失问题
- 修复BM25索引重复重建问题
- 优化systemd服务配置，Restart策略从always改为on-failure

### Fixed
- 修复compare_styles函数参数类型错误
- 修复ChromaDB路径错误（chromadb → chroma_db）
- 修复语义搜索崩溃问题，增加优雅降级处理

### Security
- 移除API密钥硬编码，使用环境变量配置
- 移除所有敏感信息和硬编码URL
