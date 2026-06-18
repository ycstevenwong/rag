"""Pending upload queue.

Anonymous UI uploads land here and wait for admin approval before being
embedded into the index. Each pending item is stored as two files in
PENDING_DIR:

    <pending_id>.<ext>   ← the original file bytes
    <pending_id>.json    ← user-supplied metadata + provenance

Admin actions:
- approve → IngestPipeline runs over the file, then both files are deleted.
- reject  → both files are deleted.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PendingItem:
    pending_id: str
    filename: str
    ext: str
    sha256: str
    uploaded_at: float
    app_code: str = ""
    tags: list[str] = field(default_factory=list)


class PendingStore:
    def __init__(self, dir: Path):
        self.dir = dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        source_path: Path,
        original_filename: str,
        app_code: str,
        tags: list[str],
    ) -> PendingItem:
        pending_id = uuid.uuid4().hex
        ext = Path(original_filename).suffix.lower()
        sha = _sha256(source_path)
        dest_file = self.dir / f"{pending_id}{ext}"
        dest_meta = self.dir / f"{pending_id}.json"
        shutil.move(str(source_path), str(dest_file))
        item = PendingItem(
            pending_id=pending_id,
            filename=original_filename,
            ext=ext,
            sha256=sha,
            uploaded_at=time.time(),
            app_code=app_code.strip(),
            tags=list(tags),
        )
        dest_meta.write_text(
            json.dumps(asdict(item), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return item

    def list(self) -> list[PendingItem]:
        items: list[PendingItem] = []
        for meta_path in self.dir.glob("*.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                items.append(PendingItem(**data))
            except Exception:
                continue
        items.sort(key=lambda i: i.uploaded_at)
        return items

    def get(self, pending_id: str) -> PendingItem | None:
        meta_path = self.dir / f"{pending_id}.json"
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return PendingItem(**data)
        except Exception:
            return None

    def file_path(self, item: PendingItem) -> Path:
        return self.dir / f"{item.pending_id}{item.ext}"

    def remove(self, pending_id: str) -> None:
        item = self.get(pending_id)
        meta_path = self.dir / f"{pending_id}.json"
        meta_path.unlink(missing_ok=True)
        if item is not None:
            self.file_path(item).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
