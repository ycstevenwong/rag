"""Flask app: upload, chat (SSE), document management."""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import (
    Flask, Response, jsonify, render_template, request, stream_with_context
)

import config as cfg
from rag.bm25_store import BM25Store
from rag.chat import ChatService
from rag.embeddings import Embedder
from rag.ingest import IngestPipeline
from rag.retriever import HybridRetriever
from rag.vector_store import FaissStore

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = cfg.FLASK_SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024

    embedder = Embedder(cfg.EMBEDDING_MODEL, cfg.EMBEDDING_BASE_URL, cfg.EMBEDDING_API_KEY)
    vector_store = FaissStore(
        vectors_path=cfg.VECTORS_PATH,
        chunks_path=cfg.CHUNKS_PATH,
        docs_path=cfg.DOCS_PATH,
    )
    bm25_store = BM25Store(cfg.BM25_PATH)
    ingest = IngestPipeline(vector_store, bm25_store, embedder)
    retriever = HybridRetriever(vector_store, bm25_store, embedder)
    chat_service = ChatService(retriever)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model=cfg.LLM_MODEL)

    @app.get("/docs")
    def list_docs():
        return jsonify([asdict(d) for d in vector_store.docs])

    @app.post("/upload")
    def upload():
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported extension {ext}"}), 400

        tmp_name = f"{uuid.uuid4().hex}{ext}"
        dest = cfg.UPLOAD_DIR / tmp_name
        original = f.filename
        f.save(dest)

        def event_stream():
            try:
                for event in ingest.ingest_stream(dest, original_filename=original):
                    yield _sse(event)
            except Exception as exc:
                yield _sse({"type": "error", "error": str(exc)})
            finally:
                dest.unlink(missing_ok=True)

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/docs/<doc_id>")
    def delete_doc(doc_id: str):
        removed = ingest.delete(doc_id)
        return jsonify({"removed_chunks": removed})

    @app.get("/chunks/<int:chunk_id>")
    def get_chunk(chunk_id: int):
        c = vector_store.get_chunk(chunk_id)
        if c is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(asdict(c))

    @app.get("/sessions/<session_id>")
    def get_session(session_id: str):
        from rag.chat import _load_session
        return jsonify(_load_session(session_id))

    @app.post("/chat")
    def chat():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or uuid.uuid4().hex
        question = (payload.get("message") or "").strip()
        if not question:
            return jsonify({"error": "empty message"}), 400

        def event_stream():
            yield _sse({"type": "session", "session_id": session_id})
            for event in chat_service.chat_stream(session_id, question):
                yield _sse(event)

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
