# MPP Project

儿童重症支原体肺炎循证 Graph RAG 网页演示。项目使用本地证据图固定证据金字塔排序、时间更新关系和六步循证过程，并可选调用 OpenAI 兼容的大模型接口生成综合回答。

## 运行环境

- Python 3.9 或更高版本
- 无必需的第三方 Python 依赖
- Chrome、Edge 或其他现代浏览器

## 启动

在仓库根目录执行：

```powershell
python web_server.py --host 127.0.0.1 --port 8765
```

然后打开：

```text
http://127.0.0.1:8765/
```

如果端口已经被占用，可以改用其他端口：

```powershell
python web_server.py --host 127.0.0.1 --port 8766
```

## 使用

1. 输入或修改临床问题。
2. 选择需要纳入的证据类型。
3. 点击“生成循证回答”。
4. 查看证据关系图、文献详情、六步循证过程和综合回答。

不启用模型时，系统仍可生成完整的本地循证报告。

如果需要连接大模型，在网页中填写 OpenAI 兼容服务的 Base URL、模型名称和 API Key，然后测试连接。API Key 只参与当前请求，不写入仓库文件。

## 项目结构

```text
MPP_Project/
├─ web_server.py              # HTTP 服务与静态页面入口
├─ web_api.py                 # 分析接口与大模型网关
├─ graph_rag_core.py          # 证据图、检索、排序和关系分析
├─ demo_graph_rag.py          # 命令行 Graph RAG 演示
├─ evidence_nodes.json        # 证据节点
├─ evidence_edges.json        # 证据关系
├─ MycoplasmaPneumonia/
│  └─ llm_client.py           # 模型 JSON 输出解析
└─ web/
   ├─ index.html
   ├─ styles.css
   └─ app.js
```

## 本地命令行演示

```powershell
python demo_graph_rag.py
```

默认使用本地规则生成报告，不需要 API Key。

## API

- `GET /api/health`：服务和证据图状态
- `GET /api/evidence`：证据节点、关系和类型
- `POST /api/model/test`：模型连接测试
- `POST /api/analyze`：执行循证检索、排序、关系分析和回答生成

## 注意

当前证据来自仓库内置的 MVP 示例数据。本项目用于展示 Graph RAG 与循证输出流程，不是经过临床验证的诊疗系统，不能替代医生判断。
