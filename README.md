# MPP EvidenceGraph

面向儿童肺炎支原体肺炎（MPP/SMPP）的本地文献混合检索与循证 Graph RAG 工作台。系统把元数据、PDF 正文、证据金字塔、发表时间和证据关系放入同一条可审计流程，输出可折叠的循证思考过程与综合回答。

本项目已核验的私有语料包含 **35,408 条源题录记录**和 **909 个 PDF 文件**；题录去重后形成 **34,074 篇唯一文献**。这些数字不能表述为“35,408 篇全文”：只有经过匹配和解析的 PDF 才属于可检索全文。

## 核心流程

1. 读取 CSV/XLSX 元数据并对 DOI、题名和来源别名去重。
2. 盘点 PDF，完成元数据匹配、重复文件识别、逐页解析和必要的本地 OCR。
3. 建立 SQLite 清单、中文 FTS5 索引、嵌入缓存和 Qdrant 本地向量集合。
4. 并行检索元数据向量、全文向量、元数据关键词和全文关键词，以 RRF 融合候选。
5. 按证据金字塔、质量、时间和全文命中情况排序，并构建 `supports`、`updates`、`supplements`、`confirms`、`conflicts`、`cautions` 六类关系。
6. 大模型先从候选原文中选择 Quote ID，并输出受控的临床维度、方向和证据角色；随后跨文献综合共识、冲突、时间更新和证据缺口。服务器校验问题焦点、引用编号、数字和可读性，失败时切换为确定性本地摘要。
7. 网页显示可折叠推理、综合文献结论、查询级证据图、来源详情、命中页码和可用的原文 PDF。

## 输出结构与证据图

每次真实知识库查询固定输出六个可折叠步骤，内容随问题和检索来源变化：

1. 检索并盘点证据库存。
2. 优先查看指南；缺少指南时明确说明。
3. 查阅系统综述，检验指南结论。
4. 聚焦关键随机对照试验，并检查时间更新。
5. 用观察性研究、叙述性综述和病例报告补充安全性与适用边界。
6. 检查各级证据一致性并形成综合判断。

综合回答采用“综合文献结论”优先结构，随后依次显示“证据链”“时间更新”“安全性与适用边界”和“证据缺口”。当文献没有直接涉及问题焦点，或所有相关声明方向都不确定时，程序不会用同疾病的其他治疗文献生成肯定或否定建议。

网页关系图只显示文献节点，并按指南、系统综述、RCT、观察性研究、叙述性综述和病例报告横向排列。六类箭头含义如下：

- `supports`：下级或汇总证据支持上级结论。
- `updates`：较新研究在相同临床维度上更新较早指南。
- `supplements`：补充不同终点、亚组、适用性或证据边界。
- `confirms`：独立证据确认相同方向的发现。
- `conflicts`：相同临床维度出现相反方向。
- `cautions`：补充安全性、特殊人群或外推限制。

为了保持第一版演示的可读性，图中最多显示 10 篇文献，并从完整关系中选取最多 12 条具有代表性的关系。所有检索来源仍保留在右侧来源列表；内部声明图和 `source_claim_ids` 继续保存每条箭头的原文溯源，不因网页节点上限而消失。点击图节点会高亮其直接关系并打开对应来源详情。

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

复制 `.env.example` 中的变量到当前 PowerShell 会话或系统环境。环境变量是持久的服务端配置选项；网页也可接受仅保留在当前页面内存中的临时 API Key。两种方式都不会将密钥写入 Git、SQLite、日志或报告：

```powershell
$env:MPP_KB_SOURCE='.\MPP'
$env:MPP_KB_DATA='.local\knowledge_base'
$env:MPP_EMBEDDING_PROVIDER='local'
$env:MPP_EMBEDDING_MODEL='intfloat/multilingual-e5-small'
$env:MPP_LOCAL_EMBEDDING_DIR='.local\models\multilingual-e5-small\onnx'
$env:MPP_API_BASE='https://provider.example/v1'
$env:MPP_API_KEY='替换为有效令牌'
$env:MPP_CHAT_MODEL='替换为兼容的聊天模型'
```

向量化默认由本机 ONNX Runtime 运行，不需要 API Key，也不会上传题名、摘要或正文。`MPP_API_BASE`、`MPP_API_KEY` 和 `MPP_CHAT_MODEL` 只用于可选的聊天大模型；API 地址必须是 OpenAI 兼容接口的 `/v1` 根地址。

## 构建知识库

先盘点语料，不写索引：

```powershell
python -m knowledge_base.cli inspect --source ".\MPP" --json
```

当前已核验输出为 `metadata_records=35408`、`pdf_files=909`、`duplicate_pdf_files=65`。`decode_replacement_count` 表示源 CSV 中需要容错解码的字符数，不等同于文献错误数。

执行免费本地阶段：

```powershell
python -m knowledge_base.cli build --json
```

该命令完成元数据导入、PDF 盘点与匹配、解析、分块和 FTS5，并返回准确的 `embedding_pending` 数量。构建状态和失败阶段可这样查看或重试：

```powershell
python -m knowledge_base.cli status --json
python -m knowledge_base.cli pause --json
python -m knowledge_base.cli retry --stage pdf_match --json
python -m knowledge_base.cli retry --stage parse --json
python -m knowledge_base.cli retry --stage chunk --json
python -m knowledge_base.cli retry --stage lexical --json
```

合法本地重试阶段为 `metadata`、`pdf_inventory`、`pdf_match`、`parse`、`chunk`、`lexical`。向量阶段还可选择 `embedding` 或 `vector`。

### 本地向量化

首次运行先下载模型文件；当前 Ryzen 9 7945HX 可使用官方 INT8 权重：

```powershell
.\.venv\Scripts\hf.exe download intfloat/multilingual-e5-small onnx/model_qint8_avx512_vnni.onnx onnx/tokenizer.json --local-dir .local\models\multilingual-e5-small
```

其他 CPU 可将文件名换为 `onnx/model.onnx`，代码会自动回退到标准权重。

网页中点击本地知识库区域的播放按钮即可开始全量向量化。任务在后台分批运行，页面显示已完成数和总数，并可在批次边界暂停或继续。

命令行也可先验证 32 条，确认模型维度和 Qdrant 集合兼容：

```powershell
python -m knowledge_base.cli build --embedding-limit 32 --confirm-embedding-cost --json
```

试批会在题录和正文之间均衡抽样，最多处理 32 条文本。执行完整本地向量化：

```powershell
python -m knowledge_base.cli build --confirm-full-embedding-cost --json
```

命令中的 `confirm` 是为兼容远程 Embedding 模式保留的旧参数名；默认 `local` 模式不会产生接口费用。重复运行会复用内容哈希缓存，不会再次处理未变化文本。

## 启动网页

```powershell
python web_server.py --host 127.0.0.1 --port 8765
```

浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。端口占用时可改为 `8766`。网页中的知识库构建按钮调用后端任务，可暂停、继续或按阶段重试；任务轮询不会阻止已经建好的索引继续回答问题。

如果本地索引尚未构建，页面会明确标记并使用仓库内置的“示例证据”，不会把示例结果伪装成真实语料检索。此模式会校验 `model_config`，但不会调用任何模型，而是返回确定性的示例输出。

### 网页“模型服务”配置

网页“模型服务”允许输入 OpenAI 兼容 API 地址、API Key 和模型名。三项必须成组填写；三项全部留空时使用服务端环境变量 `MPP_API_BASE`、`MPP_API_KEY` 和 `MPP_CHAT_MODEL`。页面填写值保留在当前页面内存中，并供刷新或离开前从该页面提交的每次查询复用；每次提交都会随该请求发送。它们不会写入文件、数据库、浏览器存储或状态 API，并在刷新或触发 `pagehide`（离开或跳转）时清空。

浏览器请求配置控制证据类型精炼、来源约束的声明抽取和跨文献结论综合，不会更改本地 E5 向量模型或重建向量。模型报告必须通过问题焦点、引用、数字和可读性校验；模型不可用或校验失败时，服务器使用确定性本地摘要。公网 API 地址必须使用 HTTPS；`localhost`、`127.0.0.1` 和 `::1` 可使用 HTTP。API Key 默认掩码显示，可切换可见性。服务商返回 401、403 或模型权限错误时，请改用有效令牌和已获授权的模型，切勿把密钥粘贴到源代码中。

1. 打开网页。
2. 在“模型服务”中成组填写 API 地址、API Key 和模型名，或全部留空以使用服务端环境变量。
3. 输入临床问题并运行证据查询。

## 标准问题验证

`validation/mpp_questions.json` 包含 12 个经过元数据核对的中文问题，覆盖激素使用、剂量与时机、大环内酯耐药、四环素类、氟喹诺酮类、支气管镜、IVIG、血栓并发症、难治/重症识别和长期结局。

完成本地索引后运行：

```powershell
python validation/run_validation.py
```

报告写入 `.local\validation\mpp-validation-时间戳.json`，统计预期来源命中、重复文献数、不受支持引用数、证据类型覆盖率和检索模式。它只建立当前基线，不预设未经人工标注验证的准确率阈值。至少人工复核 `steroid-use`、`steroid-dose` 和 `macrolide-resistance` 三项的命中片段与 PDF 页码。

如果网页显示 `query_embedding_unavailable`，表示本地模型或向量集合暂时不可用，查询会降级为关键词检索，不表示报告流程失败。当前已核验语料为 34,074 篇唯一文献、10,697 个文本块，共 44,771 条待向量化文本；本地任务写入首批向量后，查询即可升级为混合检索，无需等待全量完成。

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

私有 PDF、CSV/XLS/XLSX 元数据、ZIP、`.env`、SQLite、Qdrant、本地模型、OCR/嵌入缓存、日志和验证输出均由 `.gitignore` 排除。向别人分享代码时只提交仓库中的代码和配置示例；对方需要自行准备合法语料、下载本地向量模型并重新构建索引。只有启用聊天大模型时才需要有效模型令牌。`127.0.0.1` 只允许本机访问，局域网或公网部署还需要独立的访问控制、HTTPS、密钥管理和文献版权审查。

本地向量化不会把文献内容发送给外部服务。启用聊天大模型后，证据精炼与声明标签抽取会把本次候选来源的题名和查询命中片段发送给聊天模型，最终六步报告在本地生成。处理受版权、伦理或机构数据政策约束的材料前，应使用获准的服务商或本地部署模型，并完成数据出境与隐私评估。原始 PDF 文件本身不会作为文件上传给模型接口。

## 医疗声明

本系统是科研与汇报用途的循证决策支持原型，输出用于辅助证据检索与追溯，不是医疗器械，也不能替代儿科、呼吸科、感染科或重症医学专业人员的诊断和个体化治疗决策。
