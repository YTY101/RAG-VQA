# 基于 RAG 的图像问答

本项目按课程 PDF 中“基于 RAG 的图像问答”要求实现完整流程：输入图像和自然语言问题，系统生成图文融合查询，检索本地知识库与可选互联网证据，过滤重排后输出带支撑证据的答案。

## 功能对应

- Step 1 Query 生成：BLIP 图像描述 + 问题关键词抽取，形成文本 Query；图像本身作为视觉 Query。
- Step 2 内容检索：本地文本向量检索 + 本地图像相似检索；可选 Wikipedia 外部检索。
- Step 3 证据整理与增强：相似度排序、去重、Top-k 截断、来源与分数结构化封装。
- Step 4 回答问题：融合图像描述、视觉 VQA 结果和证据，生成答案并返回引用证据。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行会自动下载 Hugging Face 模型。如果机器算力有限，代码会回退到轻量规则与哈希向量，仍可跑通流程。

## 快速运行

构建本地知识库索引：

```bash
python -m rag_vqa.cli build-index \
  --kb data/knowledge_base/sample_knowledge.jsonl \
  --index-dir outputs/index
```

对一张图提问：

```bash
python -m rag_vqa.cli ask \
  --image /path/to/image.jpg \
  --question "这座建筑有什么历史意义？" \
  --kb data/knowledge_base/sample_knowledge.jsonl \
  --index-dir outputs/index \
  --web
```

启动可视化 Demo：

```bash
python -m rag_vqa.cli serve --web --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 知识库格式

知识库使用 JSONL，每行一个证据文档：

```json
{
  "id": "landmark_eiffel",
  "title": "埃菲尔铁塔",
  "text": "埃菲尔铁塔位于法国巴黎战神广场...",
  "source": "local_demo/wiki_summary",
  "type": "text",
  "image_path": null,
  "tags": ["建筑", "地标", "巴黎"],
  "metadata": {"language": "zh"}
}
```

如果有图库，可把 `image_path` 指向图片文件，系统会用 CLIP 或颜色直方图进行图像向量检索。

## 主要文件

- `rag_vqa/cli.py`：命令行和 Gradio 入口。
- `rag_vqa/pipeline.py`：端到端 RAG-VQAs 流水线。
- `rag_vqa/vision.py`：图像描述与视觉问答。
- `rag_vqa/query.py`：图文融合 Query 构建。
- `rag_vqa/retriever.py`：本地向量知识库与证据重排。
- `rag_vqa/web_retriever.py`：可选 Wikipedia 外部证据检索。
- `rag_vqa/answer.py`：基于证据的答案生成。
