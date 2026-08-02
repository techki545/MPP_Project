# MPP EvidenceGraph

面向儿童肺炎支原体肺炎（MPP/SMPP）的本地文献混合检索与循证 Graph RAG 工作台。系统把元数据、PDF 正文、证据金字塔、发表时间和证据关系放入同一条可审计流程，输出可折叠的循证思考过程与综合回答。

本项目已核验的私有语料包含 **35,408 条文献元数据记录**和 **909 个 PDF 文件**。这两个数字不能表述为“35,408 篇全文”：只有经过匹配和解析的 PDF 才属于可检索全文。

## 核心流程

1. 读取 CSV/XLSX 元数据并对 DOI、题名和来源别名去重。
2. 盘点 PDF，完成元数据匹配、重复文件识别、逐页解析和必要的本地 OCR。
3. 建立 SQLite 清单、中文 FTS5 索引、嵌入缓存和 Qdrant 本地向量集合。
4. 并行检索元数据向量、全文向量、元数据关键词和全文关键词，以 RRF 融合候选。
5. 按证据金字塔、质量、时间和全文命中情况排序，并构建 `supports`、`updates`、`supplements`、`conflicts`、`cautions` 关系。
6. 大模型仅在后端读取候选证据，临床数字必须能追溯到来源片段；模型不可用时返回确定性的本地报告。
7. 网页显示可折叠推理、综合回答、查询级证据图、来源详情、命中页码和可用的原文 PDF。

## 环境准备

需要 Python 3.10 及以上版本。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

可用下面的本地初始化检查确认 RapidOCR 能被加载。首次初始化可能需要数十秒：

```powershell
python -c "from rapidocr import RapidOCR; RapidOCR(); print('RapidOCR ready')"
```

复制 `.env.example` 中的变量到当前 PowerShell 会话或系统环境。密钥只放在服务端环境变量中，不写入网页、Git、SQLite、日志或验证报告：

```powershell
$env:MPP_KB_SOURCE='C:\Users\LTC\Desktop\MPP'
$env:MPP_KB_DATA='.local\knowledge_base'
$env:MPP_API_BASE='https://provider.example/v1'
$env:MPP_API_KEY='替换为有效令牌'
$env:MPP_EMBEDDING_MODEL='text-embedding-3-large'
$env:MPP_CHAT_MODEL='替换为兼容的聊天模型'
```

`MPP_API_BASE` 必须是 OpenAI 兼容接口的 `/v1` 根地址。聊天和嵌入模型都必须由当前令牌实际授权；网页中不再提供 API Key 输入框。

## 构建知识库

先盘点语料，不写索引：

```powershell
python -m knowledge_base.cli inspect --source "C:\Users\LTC\Desktop\MPP" --json
```

当前已核验输出为 `metadata_records=35408`、`pdf_files=909`、`duplicate_pdf_files=65`。`decode_replacement_count` 表示源 CSV 中需要容错解码的字符数，不等同于文献错误数。

执行免费本地阶段：

```powershell
python -m knowledge_base.cli build --json
```

该命令完成元数据导入、PDF 盘点与匹配、解析、分块和 FTS5，并在付费嵌入前停下，返回准确的 `embedding_pending` 数量。构建状态和失败阶段可这样查看或重试：

```powershell
python -m knowledge_base.cli status --json
python -m knowledge_base.cli pause --json
python -m knowledge_base.cli retry --stage pdf_match --json
python -m knowledge_base.cli retry --stage parse --json
python -m knowledge_base.cli retry --stage chunk --json
python -m knowledge_base.cli retry --stage lexical --json
```

合法本地重试阶段为 `metadata`、`pdf_inventory`、`pdf_match`、`parse`、`chunk`、`lexical`。向量阶段还可选择 `embedding` 或 `vector`，但需要费用确认。

### 付费嵌入检查点

先只验证 32 条，确认接口、模型维度和 Qdrant 集合兼容：

```powershell
python -m knowledge_base.cli build --embedding-limit 32 --confirm-embedding-cost --json
```

试批会在题录和正文之间均衡抽样，最多发送 32 条文本，并在成功写入两个向量集合后持久化模型兼容标记。检查输出不含密钥，记录剩余 `embedding_pending` 后，再由负责人进行第二次明确确认。没有成功试批标记时，完整命令会被服务端拒绝：

```powershell
python -m knowledge_base.cli build --confirm-full-embedding-cost --json
```

重复运行会复用内容哈希缓存，不会再次嵌入未变化文本。没有明确费用确认时，不应运行完整嵌入命令。

## 启动网页

```powershell
python web_server.py --host 127.0.0.1 --port 8765
```

浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。端口占用时可改为 `8766`。网页中的知识库构建按钮调用后端任务，可暂停、继续或按阶段重试；任务轮询不会阻止已经建好的索引继续回答问题。

如果本地索引尚未构建，页面会明确标记并使用仓库内置的“示例证据”，不会把示例结果伪装成真实语料检索。

## 标准问题验证

`validation/mpp_questions.json` 包含 12 个经过元数据核对的中文问题，覆盖激素使用、剂量与时机、大环内酯耐药、四环素类、氟喹诺酮类、支气管镜、IVIG、血栓并发症、难治/重症识别和长期结局。

完成本地索引后运行：

```powershell
python validation/run_validation.py
```

报告写入 `.local\validation\mpp-validation-时间戳.json`，统计预期来源命中、重复文献数、不受支持引用数、证据类型覆盖率和检索模式。它只建立当前基线，不预设未经人工标注验证的准确率阈值。至少人工复核 `steroid-use`、`steroid-dose` 和 `macrolide-resistance` 三项的命中片段与 PDF 页码。

## 命令行与兼容流程

直接查询已构建知识库：

```powershell
python -m knowledge_base.cli query "SMPP儿童是否应常规使用糖皮质激素？" --year-from 2015 --year-to 2026 --json
```

原有演示、疾病级计划和病例级执行器仍可运行：

```powershell
python demo_graph_rag.py
python -m disease.Task_level
python -m case.Case_level
```

病例结果写入已被 Git 忽略的 `case/MycoplasmaPneumonia/record/`。

## Web API

- `GET /api/health`：服务、知识库和模型状态。
- `GET /api/kb/status`：语料与索引计数。
- `GET /api/model/status`：后端模型配置状态，不返回密钥。
- `GET /api/evidence`：证据类型目录或明确标记的示例证据。
- `POST /api/query`：执行混合检索、证据图构建和双部分报告生成。
- `POST /api/kb/build`、`POST /api/kb/pause`、`POST /api/kb/retry`：后台构建控制。
- `GET /api/jobs/{job_id}`：查询后台任务。
- `GET /api/documents/{document_id}`：文献详情和命中片段。
- `GET /api/documents/{document_id}/pdf`：受约束的本地 PDF 响应。

请求体最大为 1 MB；文献 ID 经过校验，PDF 只能从已配置语料目录或已索引路径读取。

## 项目结构

```text
MPP_Project/
├─ knowledge_base/            # 导入、解析、索引、检索、图谱和报告主流程
├─ validation/                # 标准问题与可重复基线评估
├─ tests/                     # 单元、集成、API、兼容和网页测试
├─ web/                       # 交互式循证工作台
├─ disease/                   # 疾病级任务规划与本地 RAG
├─ case/                      # 病例级执行器与工具
├─ web_server.py              # HTTP 服务和静态页面入口
├─ graph_rag_core.py          # 内置示例 Graph RAG 兼容层
├─ demo_graph_rag.py          # 无密钥命令行演示
└─ .env.example               # 不含真实密钥的配置模板
```

## 数据边界与分享

私有 PDF、CSV/XLS/XLSX 元数据、ZIP、`.env`、SQLite、Qdrant、OCR/嵌入缓存、日志和验证输出均由 `.gitignore` 排除。向别人分享代码时只提交仓库中的代码和配置示例；对方需要自行准备合法语料、有效模型令牌并重新构建索引。`127.0.0.1` 只允许本机访问，局域网或公网部署还需要独立的访问控制、HTTPS、密钥管理和文献版权审查。

配置外部模型服务后，向量化会把待嵌入的题名、摘要或正文块发送给该服务；声明抽取与回答生成会把本次候选来源的题名、摘要和查询命中片段发送给聊天模型。处理受版权、伦理或机构数据政策约束的材料前，应使用获准的服务商或本地部署模型，并完成数据出境与隐私评估。原始 PDF 文件本身不会作为文件上传给模型接口。

## 医疗声明

本系统是科研与汇报用途的循证决策支持原型，输出用于辅助证据检索与追溯，不是医疗器械，也不能替代儿科、呼吸科、感染科或重症医学专业人员的诊断和个体化治疗决策。
