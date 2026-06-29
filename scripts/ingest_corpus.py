"""Bulk-ingest a directory (or single file) of documents into the RAG index.

Usage:
    python scripts/ingest_corpus.py                                 # uses data/corpus/
    python scripts/ingest_corpus.py path/to/docs/                   # custom directory
    python scripts/ingest_corpus.py path/to/file.pdf                # single file
    python scripts/ingest_corpus.py docs/manuals/ \\
        --source-type manual --tags product-x,v3.2,en               # explicit metadata
    python scripts/ingest_corpus.py data/corpus/ --by-path          # infer from folders
    python scripts/ingest_corpus.py data/corpus/manual --by-filename # manuals: filename
    python scripts/ingest_corpus.py data/corpus/spec --by-app-path  # specs: app-code folders

With --by-path, each file's source_type, app_code, version, and
functionality are derived from its folder layout:

    <target>/<source_type>/<app_code>/<version>/<functionality>/file.ext

With --by-filename, each file's functionality and version are parsed
from the filename, and source_type defaults to "manual":

    <target>/.../<functionality>_<version>_<anything>.ext

The third+ underscore-separated segments are markers and ignored. App_code
is left empty - query-time filtering on app_code uses APP_VERSION_MAP to
decide which (functionality, version) combos belong to that app, so the
manual itself doesn't need to be stamped with an app_code.

With --by-app-path, each file's app_code (and optionally version) come
from the folder layout, and source_type defaults to "spec":

    <target>/<app_code>/<file.ext>
    <target>/<app_code>/<version>/<file.ext>

Use this for app-owned docs (specs, policies) where each file belongs
to exactly one app_code and isn't covered by APP_VERSION_MAP.

Files that don't match the convention fall back to --source-type /
--app-code / --version / --functionality.

Auto-narrowing: if the target contains a 'manual/' or 'spec/' subdir
matching the chosen mode, the script narrows to that subdir before
walking. This means

    python scripts/ingest_corpus.py data/corpus --by-filename

walks only data/corpus/manual/ (not the whole corpus), and

    python scripts/ingest_corpus.py data/corpus --by-app-path

walks only data/corpus/spec/. Pointing directly at the subdir
(`data/corpus/manual`) works the same way - no double-narrowing.

Idempotent: documents already in the index (by SHA-256) are skipped.

WARNING: do NOT run this while `python app.py` is also running. Both
processes hold their own in-memory copies of the FAISS/BM25 stores and
the last writer wins on persist.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from rag.bm25_store import BM25Store
from rag.embeddings import Embedder
from rag.ingest import IngestPipeline
from rag.vector_store import FaissStore


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"
KNOWN_SOURCE_TYPES = {"manual", "spec", "other"}
VERSION_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in ALLOWED_EXTENSIONS else []
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        )
    return []


def infer_from_path(
    file_path: Path, root: Path
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (source_type, app_code, version, functionality) from
    <root>/<source_type>/<app_code>/<version>/<functionality>/.../file."""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return (None, None, None, None)
    parts = rel.parts
    source_type = parts[0].strip().lower() if len(parts) >= 2 else None
    app_code = parts[1].strip() if len(parts) >= 3 else None
    version = parts[2].strip() if len(parts) >= 4 else None
    functionality = parts[3].strip() if len(parts) >= 5 else None
    return (source_type, app_code, version, functionality)


def infer_from_filename(file_path: Path) -> tuple[str | None, str | None]:
    """Parse '<functionality>_<version>_<anything>.ext' from the filename
    (excluding extension). Return (functionality, version). Third and later
    segments are markers and ignored. version must look like 'v\\d+' or it's
    treated as missing."""
    parts = file_path.stem.split("_")
    if not parts or not parts[0]:
        return (None, None)
    functionality = parts[0].strip()
    version = None
    if len(parts) >= 2 and VERSION_RE.match(parts[1].strip()):
        version = parts[1].strip().lower()
    return (functionality, version)


def infer_from_app_path(
    file_path: Path, root: Path
) -> tuple[str | None, str | None]:
    """Return (app_code, version) from <root>/<app_code>/[<version>/]/.../file."""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return (None, None)
    parts = rel.parts
    app_code = parts[0].strip() if len(parts) >= 2 else None
    version = None
    if len(parts) >= 3 and VERSION_RE.match(parts[1].strip()):
        version = parts[1].strip().lower()
    return (app_code, version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-ingest documents into the RAG index.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_CORPUS_DIR),
        help="File or directory of documents (default: data/corpus/)",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help='Doc category applied to every file: manual | spec | other (default: "manual" with --by-filename, else "other")',
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags applied to every file (e.g. 'product-x,v3.2,en')",
    )
    parser.add_argument(
        "--app-code",
        default="",
        help="App code applied to every file (must match a value in config.APP_CODES if filtering by it)",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Version applied to every file (e.g. 'v1', 'v2')",
    )
    parser.add_argument(
        "--functionality",
        default="",
        help="Functionality applied to every file (e.g. 'login', 'token'). Used with APP_VERSION_MAP at query time.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--by-path",
        action="store_true",
        help="Infer all four fields per file from <target>/<source_type>/<app_code>/<version>/<functionality>/...",
    )
    mode.add_argument(
        "--by-filename",
        action="store_true",
        help="Parse functionality and version from '<func>_<ver>_<anything>.ext'. source_type defaults to 'manual'. app_code stays empty.",
    )
    mode.add_argument(
        "--by-app-path",
        action="store_true",
        help="Read app_code (and optional version) from <target>/<app_code>/[<version>/]/.... source_type defaults to 'spec'.",
    )
    args = parser.parse_args()

    target = Path(args.path).resolve()
    # Auto-narrow when the corpus convention is recognized so the mode only
    # walks its own subtree. Prevents `data/corpus --by-filename` from picking
    # up spec/ files and tagging them as manuals, and the reverse for specs.
    if target.is_dir():
        if args.by_filename and (target / "manual").is_dir():
            target = target / "manual"
            print(f"--by-filename: narrowed target to {target}\n")
        elif args.by_app_path and (target / "spec").is_dir():
            target = target / "spec"
            print(f"--by-app-path: narrowed target to {target}\n")
    if args.source_type is not None:
        default_source_type = args.source_type.strip().lower()
    elif args.by_filename:
        default_source_type = "manual"
    elif args.by_app_path:
        default_source_type = "spec"
    else:
        default_source_type = "other"
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    default_app_code = args.app_code.strip()
    default_version = args.version.strip()
    default_functionality = args.functionality.strip()

    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        if target == DEFAULT_CORPUS_DIR:
            print(
                f"Create {DEFAULT_CORPUS_DIR} and put documents there, "
                "or pass a path argument.",
                file=sys.stderr,
            )
        return 1

    files = collect_files(target)
    if not files:
        print(f"No supported documents found under {target}", file=sys.stderr)
        return 1

    embedder = Embedder(cfg.EMBEDDING_MODEL, cfg.EMBEDDING_BASE_URL, cfg.EMBEDDING_API_KEY)
    vector_store = FaissStore(
        vectors_path=cfg.VECTORS_PATH,
        chunks_path=cfg.CHUNKS_PATH,
        docs_path=cfg.DOCS_PATH,
        files_path=cfg.FILES_PATH,
    )
    bm25_store = BM25Store(cfg.BM25_PATH)
    ingest = IngestPipeline(vector_store, bm25_store, embedder)

    print(f"Ingesting {len(files)} document(s) from {target}\n")
    n_added = n_dup = n_err = 0

    for i, file_path in enumerate(files, 1):
        label = (
            file_path.relative_to(target) if target.is_dir() else file_path.name
        )
        print(f"[{i}/{len(files)}] {label}")

        source_type = default_source_type
        app_code = default_app_code
        version = default_version
        functionality = default_functionality
        if args.by_path and target.is_dir():
            inf_st, inf_ac, inf_ver, inf_fn = infer_from_path(file_path, target)
            if inf_st:
                source_type = inf_st
                if source_type not in KNOWN_SOURCE_TYPES:
                    print(f"  WARNING: inferred source_type {source_type!r} not in {sorted(KNOWN_SOURCE_TYPES)}")
            if inf_ac:
                app_code = inf_ac
                if cfg.APP_CODES and app_code not in cfg.APP_CODES:
                    print(f"  WARNING: inferred app_code {app_code!r} not in config.APP_CODES")
            if inf_ver:
                version = inf_ver
            if inf_fn:
                functionality = inf_fn
            print(f"  source_type={source_type} app_code={app_code or '-'} version={version or '-'} functionality={functionality or '-'}")
        elif args.by_filename:
            inf_fn, inf_ver = infer_from_filename(file_path)
            if inf_fn:
                functionality = inf_fn
            if inf_ver:
                version = inf_ver
            else:
                print(f"  WARNING: no '_v<N>_' segment in filename {file_path.name!r}; version stays {version or '-'}")
            print(f"  source_type={source_type} app_code={app_code or '-'} version={version or '-'} functionality={functionality or '-'}")
        elif args.by_app_path and target.is_dir():
            inf_ac, inf_ver = infer_from_app_path(file_path, target)
            if inf_ac:
                app_code = inf_ac
                if cfg.APP_CODES and app_code not in cfg.APP_CODES:
                    print(f"  WARNING: inferred app_code {app_code!r} not in config.APP_CODES")
            if inf_ver:
                version = inf_ver
            print(f"  source_type={source_type} app_code={app_code or '-'} version={version or '-'} functionality={functionality or '-'}")

        try:
            for event in ingest.ingest_stream(
                file_path,
                original_filename=file_path.name,
                source_type=source_type,
                tags=tags,
                app_code=app_code,
                version=version,
                functionality=functionality,
                managed=True,
            ):
                t = event["type"]
                if t == "stage":
                    print(f"  {event['stage']}...")
                elif t == "progress":
                    sys.stdout.write(f"\r  Embedding batch {event['done']}/{event['total']}")
                    sys.stdout.flush()
                    if event["done"] == event["total"]:
                        print()
                elif t == "done":
                    r = event["result"]
                    if r.get("duplicate"):
                        print("  Already indexed -> skipped")
                        n_dup += 1
                    else:
                        print(f"  Indexed {r['n_chunks']} chunks")
                        n_added += 1
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            n_err += 1

    print(f"\nDone. Added: {n_added}  Duplicate: {n_dup}  Errors: {n_err}")
    print(f"Total docs in index: {len(vector_store.docs)}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
