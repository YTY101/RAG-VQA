from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .debug import debug_dump
from .pipeline import RAGVQAPipeline
from .retriever import KnowledgeBase


def build_index(args: argparse.Namespace) -> None:
    settings = Settings(debug=args.debug)
    debug_dump(settings, "cli.build_index.args", vars(args))
    kb = KnowledgeBase.from_jsonl(args.kb, settings=settings)
    kb.save(args.index_dir)
    print(f"Built index with {len(kb.docs)} documents: {args.index_dir}")


def ask(args: argparse.Namespace) -> None:
    settings = Settings(top_k=args.top_k, debug=args.debug)
    debug_dump(settings, "cli.ask.args", vars(args))
    index_dir = Path(args.index_dir)
    if index_dir.exists() and (index_dir / "documents.json").exists():
        kb = KnowledgeBase.load(index_dir, settings=settings)
        debug_dump(
            settings,
            "index.load",
            {
                "index_dir": str(index_dir),
                "doc_count": len(kb.docs),
                "text_vector_shape": kb.text_vectors.shape,
                "image_vector_shape": kb.image_vectors.shape,
            },
        )
    else:
        kb = KnowledgeBase.from_jsonl(args.kb, settings=settings)
        kb.save(index_dir)

    pipeline = RAGVQAPipeline(kb=kb, settings=settings, enable_web=args.web)
    result = pipeline.ask(args.image, args.question, top_k=args.top_k)
    payload = asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def serve(args: argparse.Namespace) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise SystemExit("Please install gradio first: pip install gradio") from exc

    settings = Settings(top_k=args.top_k, debug=args.debug)
    debug_dump(settings, "cli.serve.args", vars(args))
    index_dir = Path(args.index_dir)
    kb = KnowledgeBase.load(index_dir, settings=settings) if (index_dir / "documents.json").exists() else KnowledgeBase.from_jsonl(args.kb, settings)
    pipeline = RAGVQAPipeline(kb=kb, settings=settings, enable_web=args.web)

    def infer(image, question):
        result = pipeline.ask(image, question, top_k=args.top_k)
        evidence_lines = [
            f"[{i}] {ev.title} | score={ev.score:.3f} | {ev.source}\n{ev.content}"
            for i, ev in enumerate(result.evidences, start=1)
        ]
        return result.answer, result.visual_caption, "\n\n".join(evidence_lines)

    demo = gr.Interface(
        fn=infer,
        inputs=[gr.Image(type="filepath", label="图像"), gr.Textbox(label="问题")],
        outputs=[gr.Textbox(label="答案"), gr.Textbox(label="图像描述"), gr.Textbox(label="支撑证据")],
        title="基于 RAG 的图像问答",
    )
    demo.launch(server_name=args.host, server_port=args.port)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-based visual question answering")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build local vector index")
    p_build.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_build.add_argument("--index-dir", default="outputs/index")
    p_build.add_argument("--debug", action="store_true", help="Print intermediate variables to stderr")
    p_build.set_defaults(func=build_index)

    p_ask = sub.add_parser("ask", help="Ask a question about an image")
    p_ask.add_argument("--image", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_ask.add_argument("--index-dir", default="outputs/index")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--web", action="store_true", help="Enable Wikipedia evidence retrieval")
    p_ask.add_argument("--debug", action="store_true", help="Print intermediate variables to stderr")
    p_ask.set_defaults(func=ask)

    p_serve = sub.add_parser("serve", help="Run a Gradio demo")
    p_serve.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_serve.add_argument("--index-dir", default="outputs/index")
    p_serve.add_argument("--top-k", type=int, default=5)
    p_serve.add_argument("--web", action="store_true")
    p_serve.add_argument("--debug", action="store_true", help="Print intermediate variables to stderr")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.set_defaults(func=serve)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
