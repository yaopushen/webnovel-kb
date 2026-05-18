# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.1] - 2026-05-18

### Changed
- 日志优化，文件轮转、分级配置、结构化输出
- 移除 rank_bm25 + FAISS，纯 Tantivy+ChromaDB 
- 为智能搜索加入网络搜索工具

## [1.8] - 2026-05-17

### Added
- 新增 `get_chapter_edges` 工具，用于提取章节开头和结尾段落
- 新增 `list_chapters` 和 `read_chapter` 工具，支持章节级别的浏览
- 新增章节解析支持，兼容裸数字章节号（如 001、002）
- 新增详细的 MCP 工具参数文档

### Changed
- 全面更新 MAINTENANCE.md 文档，包含 11 个工具的完整参数说明
- 优化章节标题识别正则表达式
- 更新项目历史记录到 v1.8

### Fixed
- 修复 `smart_search` 工具描述，与实际实现保持一致
- 修复知识库统计和风格分析工具的合并问题

## [1.7] - 2026-05-15

### Added
- 新增 OAuth 2.0 PKCE 内置授权服务器（`oauth_auth.py`）
- 新增 `/authorize`、`/token`、`/well-known/oauth-authorization-server` 端点
- 新增 `_OAuthFastMCP` 类，子类化 FastMCP 注入 OAuth 路由
- 新增 JWT (HS256) Token 签发和验证
- 新增 PKCE S256 challenge 验证机制
- 新增 `MCP_OAUTH_ISSUER_URL`、`OAUTH_JWT_SECRET`、`OAUTH_TOKEN_EXPIRY` 环境变量

### Changed
- **重大变更**：传输层从 SSE 改为 Streamable-HTTP
- 解决多 session 冲突导致返回 null 的问题
- 每次请求独立 HTTP POST，不再维护持久连接
- 内网无需认证，公网通过 PKCE 流程保护
- 更新 systemd 服务配置，旧服务 `webnovel-kb` 已 mask

### Security
- 实现 Bearer Token 验证中间件
- 实现 Token 有效期管理（默认 24 小时）
- 实现授权码 60 秒有效期限制

## [1.6] - 2026-05-11

### Added
- 新增 `smart_search` 智能搜索工具
- 基于 Chat 模型的函数调用模式搜索
- 支持多轮迭代（最多 200 轮）和并行工具调用
- 支持自动降级到 hybrid 搜索

### Changed
- 工具精简：从 15 个工具减少到 10 个工具
- 移除管理类工具（`ingest_novel`、`extract`、`get_task_status`）
- 合并 `stats` 和 `style_analysis` 功能
- 优化工具架构，专注搜索和分析功能

### Fixed
- 修复工具参数验证问题
- 优化 LLM 调用失败时的降级策略

## [1.5] - 2026-04-30

### Added
- 新增统一搜索接口，支持 semantic、bm25、hybrid、rerank 四种模式
- 新增实体语义搜索功能
- 新增知识库搜索功能（情节模式、写作模板）
- 新增异步提取任务支持
- 新增 MCP 工具集，包含 12 个工具

### Changed
- 重构为模块化架构，分离核心逻辑、搜索、提取、分析等模块
- 优化内存使用，嵌入缓存从 Python list 改为 numpy.float32，内存占用减少约 80%
- 修复语义搜索维度不匹配问题（ChromaDB 默认嵌入函数维度与 API 模型维度不一致）
- 修复 API 客户端 URL 路径缺失问题
- 修复 BM25 索引重复重建问题
- 优化 systemd 服务配置，Restart 策略从 always 改为 on-failure，避免内存泄漏

### Fixed
- 修复 compare_styles 函数参数类型错误
- 修复 ChromaDB 路径错误（chromadb → chroma_db）
- 修复语义搜索崩溃问题，增加优雅降级处理

### Security
- 移除 API 密钥硬编码，使用环境变量配置

## [1.0] - 2025-04-30

### Added
- 初始版本发布
- 单体架构实现（server.py 2200+ 行）
- 基础的小说导入和分块功能
- 基本的语义搜索功能
- ChromaDB 向量数据库集成
- MCP 服务器基础框架
