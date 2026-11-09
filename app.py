"""Flask app: upload, chat (SSE), document management."""
from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from flask import (
    Flask, Response, jsonify, render_template, request, session, stream_with_context
)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash

import config as cfg
from rag.bm25_store import BM25Store
from rag.chat import ChatService
from rag.embeddings import Embedder
from rag.ingest import IngestPipeline
from rag.pending import PendingStore
from rag.retriever import HybridRetriever
from rag.vector_store import FaissStore
from rag.writer_lock import acquire_writer_lock

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}

# These characters map to a file name unchanged, so two IDs can't share a file.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Login rate limit — per remote IP, in-memory, cleared on success.
_LOGIN_WINDOW_SECONDS = 300   # 5 minutes
_LOGIN_MAX_FAILURES = 5
_login_failures: dict[str, list[float]] = {}


def _login_allowed(ip: str) -> bool:
    now = time.time()
    fails = [t for t in _login_failures.get(ip, []) if now - t < _LOGIN_WINDOW_SECONDS]
    _login_failures[ip] = fails
    return len(fails) < _LOGIN_MAX_FAILURES


def _record_login_failure(ip: str) -> None:
    now = time.time()
    # Forget addresses whose failures have all expired, so the dict can't grow forever.
    for known in list(_login_failures):
        if all(now - t >= _LOGIN_WINDOW_SECONDS for t in _login_failures[known]):
            del _login_failures[known]
    _login_failures.setdefault(ip, []).append(now)


def _credentials_match(username, password) -> bool:
    """Always runs the slow password hash, so a wrong username takes as long to
    reject as a wrong password and valid usernames can't be found by timing."""
    if not isinstance(username, str) or not isinstance(password, str):
        return False
    password_ok = check_password_hash(cfg.ADMIN_PASSWORD_HASH, password)
    username_ok = hmac.compare_digest(username.encode(), cfg.ADMIN_USERNAME.encode())
    return username_ok and password_ok


def _is_admin() -> bool:
    return _admin_enabled() and bool(session.get("is_admin"))


def _admin_enabled() -> bool:
    return bool(cfg.ADMIN_USERNAME and cfg.ADMIN_PASSWORD_HASH)


def _free_disk_gb() -> float:
    try:
        return shutil.disk_usage(str(cfg.DATA_DIR)).free / (1024 ** 3)
    except OSError:
        return float("inf")


def _pending_total_bytes() -> int:
    total = 0
    try:
        for p in cfg.PENDING_DIR.iterdir():
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _cleanup_orphan_uploads() -> int:
    """Remove leftover files in data/uploads/ older than UPLOAD_ORPHAN_MAX_AGE.
    These come from ingests that crashed before the move/unlink step. Returns
    the number of files removed."""
    cutoff = time.time() - cfg.UPLOAD_ORPHAN_MAX_AGE
    removed = 0
    try:
        for f in cfg.UPLOAD_DIR.iterdir():
            # Uploads have generated names; dotfiles like .gitkeep aren't uploads.
            if not f.is_file() or f.name.startswith("."):
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def _is_reloader_parent() -> bool:
    """With FLASK_DEBUG, Werkzeug's parent process only watches files and a child
    serves requests, so the parent must not take the writer lock."""
    return cfg.FLASK_DEBUG and os.environ.get("WERKZEUG_RUN_MAIN") != "true"


def create_app() -> Flask:
    if _admin_enabled() and cfg.FLASK_SECRET_KEY in cfg.INSECURE_SECRET_KEYS:
        raise RuntimeError(
            "Admin login is configured but FLASK_SECRET_KEY is a published default, "
            "so anyone could forge an admin session. Set FLASK_SECRET_KEY to a long "
            "random value in .env."
        )
    if not _is_reloader_parent():
        acquire_writer_lock(cfg.INDEX_DIR / ".lock")
    app = Flask(__name__, static_folder="static", template_folder="templates")
    if cfg.TRUSTED_PROXY_COUNT:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=cfg.TRUSTED_PROXY_COUNT)
    app.config["SECRET_KEY"] = cfg.FLASK_SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = cfg.SESSION_COOKIE_SECURE
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=cfg.SESSION_LIFETIME_HOURS)

    embedder = Embedder(cfg.EMBEDDING_MODEL, cfg.EMBEDDING_BASE_URL, cfg.EMBEDDING_API_KEY)
    vector_store = FaissStore(
        vectors_path=cfg.VECTORS_PATH,
        chunks_path=cfg.CHUNKS_PATH,
        docs_path=cfg.DOCS_PATH,
        files_path=cfg.FILES_PATH,
    )
    bm25_store = BM25Store.from_chunks(vector_store.chunks)
    ingest = IngestPipeline(vector_store, bm25_store, embedder)
    pending_store = PendingStore(cfg.PENDING_DIR)
    retriever = HybridRetriever(vector_store, bm25_store, embedder)
    chat_service = ChatService(retriever)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            model=cfg.LLM_MODEL,
            app_codes=cfg.APP_CODES,
        )

    @app.get("/docs")
    def list_docs():
        items = []
        managed_count = 0
        admin = _is_admin()
        for d in vector_store.docs:
            if d.managed:
                managed_count += 1
                if not admin:
                    continue
            f = vector_store.get_file(d.file_id)
            entry = asdict(d)
            if f is not None:
                entry["filename"] = d.filename or f.filename
                entry["n_chunks"] = f.n_chunks
                entry["uploaded_at"] = f.uploaded_at
            else:
                entry["filename"] = "(missing file)"
                entry["n_chunks"] = 0
                entry["uploaded_at"] = 0.0
            items.append(entry)
        return jsonify({"items": items, "managed_count": managed_count})

    @app.post("/upload")
    def upload():
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            hint = ""
            if ext == ".doc":
                hint = " Open the file in Word and save as .docx, then re-upload."
            elif ext == ".rtf":
                hint = " Convert to .docx or .txt first."
            elif ext in (".xls", ".ppt"):
                modern = ".xlsx" if ext == ".xls" else ".pptx"
                hint = f" Open the file and save as {modern}, then re-upload."
            supported = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS))
            return jsonify({
                "error": f"Unsupported extension {ext}. Supported: {supported}.{hint}",
            }), 400

        # Pre-flight storage checks. Return 507 (Insufficient Storage) with a
        # clean error rather than letting the write raise Errno 28 mid-stream.
        if _free_disk_gb() < cfg.MIN_FREE_DISK_GB:
            return jsonify({"error": "Server storage is low. Try again later."}), 507
        if not _is_admin():
            estimated_size = request.content_length or 0
            if _pending_total_bytes() + estimated_size > cfg.PENDING_MAX_MB * 1024 * 1024:
                return jsonify({
                    "error": "Approval queue is full. Admin must process pending uploads before more can be accepted.",
                }), 507

        tmp_name = f"{uuid.uuid4().hex}{ext}"
        dest = cfg.UPLOAD_DIR / tmp_name
        original = f.filename
        raw_tags = request.form.get("tags") or ""
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        app_code = (request.form.get("app_code") or "").strip()
        source_type = (request.form.get("source_type") or "").strip().lower()
        version = (request.form.get("version") or "").strip()
        functionality = (request.form.get("functionality") or "").strip()
        requester = (request.form.get("requester") or "").strip()
        # Manuals do not carry an app_code — they're scoped by APP_VERSION_MAP
        # at query time via (app_code, functionality), so the doc itself stays
        # universal.
        if source_type == "manual":
            app_code = ""
        f.save(dest)

        # Anonymous uploads go to the pending queue; admin uploads ingest now.
        if not _is_admin():
            item = pending_store.add(
                dest,
                original,
                app_code=app_code,
                tags=tags,
                source_type=source_type or "other",
                version=version,
                functionality=functionality,
                requester=requester,
            )

            def event_stream():
                yield _sse({
                    "type": "queued",
                    "pending_id": item.pending_id,
                    "filename": item.filename,
                })

            return Response(
                stream_with_context(event_stream()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Admin path uses the same source_type/version/functionality but defaults
        # source_type to "other" if blank (anonymous uses the form's hint).
        if not source_type:
            source_type = "other"

        def event_stream():
            try:
                for event in ingest.ingest_stream(
                    dest,
                    original_filename=original,
                    source_type=source_type,
                    tags=tags,
                    app_code=app_code,
                    version=version,
                    functionality=functionality,
                    managed=True,
                ):
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
        doc = next((d for d in vector_store.docs if d.doc_id == doc_id), None)
        if doc and doc.managed and not _is_admin():
            return jsonify({"error": "Managed document; admin login required."}), 403
        removed = ingest.delete(doc_id)
        return jsonify({"removed_chunks": removed})

    @app.patch("/docs/<doc_id>")
    def update_doc(doc_id: str):
        if not _is_admin():
            return jsonify({"error": "Admin only"}), 403
        doc = next((d for d in vector_store.docs if d.doc_id == doc_id), None)
        if doc is None:
            return jsonify({"error": "Not found"}), 404
        payload = request.get_json(silent=True) or {}
        if "source_type" in payload:
            doc.source_type = (str(payload["source_type"]).strip().lower() or "other")
        if "app_code" in payload:
            doc.app_code = str(payload["app_code"]).strip()
        if "version" in payload:
            doc.version = str(payload["version"]).strip()
        if "functionality" in payload:
            doc.functionality = str(payload["functionality"]).strip()
        if "tags" in payload:
            raw_tags = payload["tags"]
            if isinstance(raw_tags, str):
                doc.tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            else:
                doc.tags = [t for t in raw_tags if isinstance(t, str) and t.strip()]
        # Manuals can't carry an app_code regardless of what was sent.
        if doc.source_type == "manual":
            doc.app_code = ""
        vector_store.persist()
        return jsonify({"updated": doc_id})

    @app.get("/admin/me")
    def admin_me():
        return jsonify({
            "enabled": _admin_enabled(),
            "is_admin": _is_admin(),
        })

    @app.post("/admin/login")
    def admin_login():
        if not _admin_enabled():
            return jsonify({"error": "Admin login is not configured."}), 503
        ip = request.remote_addr or "unknown"
        if not _login_allowed(ip):
            return jsonify({"error": "Too many failed attempts. Try again in a few minutes."}), 429
        payload = request.get_json(force=True) or {}
        if not isinstance(payload, dict):
            payload = {}
        username = payload.get("username") or ""
        if isinstance(username, str):
            username = username.strip()
        if not _credentials_match(username, payload.get("password") or ""):
            _record_login_failure(ip)
            return jsonify({"error": "Invalid username or password."}), 401
        _login_failures.pop(ip, None)
        session.permanent = True
        session["is_admin"] = True
        return jsonify({"is_admin": True})

    @app.post("/admin/logout")
    def admin_logout():
        session.pop("is_admin", None)
        return jsonify({"is_admin": False})

    @app.get("/admin/pending")
    def admin_pending_list():
        if not _is_admin():
            return jsonify({"error": "Admin only"}), 403
        items = [asdict(i) for i in pending_store.list()]
        return jsonify({"items": items})

    @app.post("/admin/pending/<pending_id>/reject")
    def admin_pending_reject(pending_id: str):
        if not _is_admin():
            return jsonify({"error": "Admin only"}), 403
        item = pending_store.get(pending_id)
        if item is None:
            return jsonify({"error": "Not found"}), 404
        pending_store.remove(pending_id)
        return jsonify({"rejected": pending_id})

    @app.post("/admin/pending/<pending_id>/approve")
    def admin_pending_approve(pending_id: str):
        if not _is_admin():
            return jsonify({"error": "Admin only"}), 403
        item = pending_store.get(pending_id)
        if item is None:
            return jsonify({"error": "Not found"}), 404
        payload = request.get_json(silent=True) or {}
        # Admin may override every value, but defaults to what the requester suggested.
        source_type = (payload.get("source_type") or item.source_type or "other").strip().lower()
        version = (payload.get("version") or item.version or "").strip()
        functionality = (payload.get("functionality") or item.functionality or "").strip()
        app_code = (payload.get("app_code") or item.app_code or "").strip()
        if source_type == "manual":
            app_code = ""
        raw_tags = payload.get("tags")
        if raw_tags is None:
            tags = list(item.tags or [])
        elif isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        else:
            tags = [t for t in raw_tags if isinstance(t, str) and t.strip()]
        file_path = pending_store.file_path(item)

        def event_stream():
            try:
                for event in ingest.ingest_stream(
                    file_path,
                    original_filename=item.filename,
                    source_type=source_type,
                    tags=tags,
                    app_code=app_code,
                    version=version,
                    functionality=functionality,
                    managed=True,
                ):
                    # Only a finished ingest removes the item. Errors and client
                    # disconnects leave it queued so the admin can retry or reject.
                    if event["type"] == "done":
                        pending_store.remove(pending_id)
                    yield _sse(event)
            except Exception as exc:
                yield _sse({"type": "error", "error": str(exc)})

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/chunks/<int:chunk_id>")
    def get_chunk(chunk_id: int):
        if not _is_admin():
            return jsonify({"error": "Admin only"}), 403
        c = vector_store.get_chunk(chunk_id)
        if c is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(asdict(c))

    @app.get("/sessions/<session_id>")
    def get_session(session_id: str):
        if not _is_admin():
            return jsonify({"error": "Admin only"}), 403
        from rag.chat import _load_session
        return jsonify(_load_session(session_id))

    @app.post("/chat")
    def chat():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or uuid.uuid4().hex
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            return jsonify({"error": "Invalid session_id"}), 400
        question = (payload.get("message") or "").strip()
        if not question:
            return jsonify({"error": "empty message"}), 400

        raw_filters = payload.get("filters") or {}
        filters: dict | None = None
        if isinstance(raw_filters, dict):
            st = (raw_filters.get("source_type") or "").strip().lower()
            ac = (raw_filters.get("app_code") or "").strip()
            tags = [t.strip() for t in (raw_filters.get("tags") or []) if isinstance(t, str) and t.strip()]
            if st or ac or tags:
                filters = {}
                if st:
                    filters["source_type"] = st
                if ac:
                    filters["app_code"] = ac
                if tags:
                    filters["tags"] = tags

        def event_stream():
            yield _sse({"type": "session", "session_id": session_id})
            for event in chat_service.chat_stream(session_id, question, filters=filters):
                yield _sse(event)

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    removed = _cleanup_orphan_uploads()
    if removed:
        print(f"[startup] cleaned {removed} orphan upload file(s) from {cfg.UPLOAD_DIR}", flush=True)
    removed = pending_store.cleanup_orphans(cfg.UPLOAD_ORPHAN_MAX_AGE)
    if removed:
        print(f"[startup] cleaned {removed} orphan pending file(s) from {cfg.PENDING_DIR}", flush=True)

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=cfg.FLASK_DEBUG, threaded=True)
