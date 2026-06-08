from rag.chunker import chunk_blocks, count_tokens
from rag.parsers import Block


def test_groups_small_blocks_into_chunks():
    blocks = [Block(text="alpha beta gamma " * 5, kind="paragraph", meta={"page": 1}) for _ in range(6)]
    chunks = chunk_blocks(blocks, target_tokens=120, overlap_tokens=0)
    assert chunks
    for c in chunks:
        assert c.token_count <= 140


def test_respects_page_boundary():
    blocks = [
        Block(text="page one content", kind="paragraph", meta={"page": 1}),
        Block(text="page two content", kind="paragraph", meta={"page": 2}),
    ]
    chunks = chunk_blocks(blocks, target_tokens=1000, overlap_tokens=0)
    assert len(chunks) == 2
    assert chunks[0].meta.get("pages") == [1]
    assert chunks[1].meta.get("pages") == [2]


def test_oversized_block_is_split():
    huge_text = "word " * 2000
    blocks = [Block(text=huge_text, kind="paragraph", meta={"page": 1})]
    chunks = chunk_blocks(blocks, target_tokens=200, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 240


def test_heading_path_carried():
    blocks = [
        Block(text="Setup", kind="heading", meta={"heading_path": "Setup", "level": 1}),
        Block(text="Install via pip.", kind="paragraph", meta={"heading_path": "Setup"}),
    ]
    chunks = chunk_blocks(blocks, target_tokens=1000, overlap_tokens=0)
    assert chunks[0].meta.get("heading_path") == "Setup"
