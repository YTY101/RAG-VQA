# 课程项目报告：基于 RAG 的图像问答

## 1. 任务目标

本项目面向“基于 RAG 的图像问答”任务：给定一张图像和一个自然语言问题，系统不仅依赖视觉模型识别图像内容，还会从外部知识库或互联网检索相关证据，并基于“图像 + 问题 + 证据”生成可解释答案。

相比传统 VQA 直接回答，本项目强调证据约束，降低幻觉回答，适合建筑历史、人物身份、地点背景、产品参数等需要外部知识支撑的问题。

## 2. 方法设计

### Step 1：Query 生成

系统首先使用图像描述模型生成图像全局语义，例如“a tower in Paris”或“an indoor room with fire extinguisher”。随后将图像描述与用户问题融合，抽取关键词并生成文本 Query。同时，原始图像会被编码为视觉 Query，用于相似图像检索。

对应代码：`rag_vqa/vision.py`、`rag_vqa/query.py`。

### Step 2：内容检索

检索分为两路：

- 文本检索：使用 SentenceTransformer 生成文本向量，在本地知识库中计算余弦相似度。
- 图像检索：使用 CLIP 图像编码器对查询图片和知识库图片编码，召回视觉相似图片及其元数据。

此外，命令行传入 `--web` 时会启用 Wikipedia 检索，补充外部百科证据。

对应代码：`rag_vqa/retriever.py`、`rag_vqa/web_retriever.py`。

### Step 3：证据整理与增强

系统对召回结果进行融合排序，并执行去重、低分过滤、Top-k 截断。每条证据都包含来源、类型、分数和内容片段，便于最终答案引用和人工检查。

输出格式示例：

```text
[1] 埃菲尔铁塔 | score=0.83 | local_demo/wiki_summary
埃菲尔铁塔位于法国巴黎战神广场，建成于1889年...
```

对应代码：`rag_vqa/retriever.py` 的 `retrieve` 与 `_snippet`。

### Step 4：回答生成

最终回答模块接收图像描述、视觉 VQA 直接答案、Top-k 证据和原始问题。若本地生成模型可用，则使用文本生成模型生成答案；否则使用可解释的抽取式回退策略，保证在低资源环境下仍能输出答案与证据来源。

对应代码：`rag_vqa/answer.py`、`rag_vqa/pipeline.py`。

## 3. 创新点

- 双路检索：文本 Query 与图像 Query 同时参与证据召回。
- 可解释输出：答案之外返回证据内容、来源和相似度分数。
- 低资源可运行：模型下载失败或无 GPU 时，自动回退到哈希文本向量、颜色直方图图像向量与抽取式回答。s
- 可扩展知识库：支持 JSONL 追加本地文本、图片路径和元数据，也支持可选互联网检索。

## 4. 使用方式

```bash
pip install -r requirements.txt
python -m rag_vqa.cli build-index --kb data/knowledge_base/sample_knowledge.jsonl --index-dir outputs/index
python -m rag_vqa.cli ask --image /path/to/image.jpg --question "这座建筑有什么历史意义？" --web
```

Web Demo：

```bash
python -m rag_vqa.cli serve --web --port 7860
```

## 5. 结果分析建议

实验时可准备三类问题：

- 纯视觉问题：如“图中有几个红色物体？”
- 视觉 + 常识问题：如“图中的设备通常用于什么场景？”
- 视觉 + 外部知识问题：如“这座建筑有什么历史意义？”

建议记录每类问题的答案准确率、证据相关性、是否出现幻觉回答，并对比开启 `--web` 前后的结果变化。
