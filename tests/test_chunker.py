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


def _overlapped(*blocks):
    return chunk_blocks(list(blocks), target_tokens=600, overlap_tokens=80)


def test_overlap_does_not_cross_a_section_change():
    chunks = _overlapped(
        Block("Limit is 500.", "paragraph", {"page": 1, "heading_path": "Limits"}),
        Block("Timeout is 30.", "paragraph", {"page": 1, "heading_path": "Timeouts"}),
    )
    assert chunks[1].text == "Timeout is 30."
    assert chunks[1].meta == {"pages": [1], "heading_path": "Timeouts"}


def test_overlap_across_a_page_break_widens_the_cited_pages():
    chunks = _overlapped(
        Block("End of page one.", "paragraph", {"page": 1, "heading_path": "Limits"}),
        Block("Start of page two.", "paragraph", {"page": 2, "heading_path": "Limits"}),
    )
    assert chunks[1].text == "End of page one.\n\nStart of page two."
    assert chunks[1].meta["pages"] == [1, 2]


def test_overlap_does_not_cross_a_slide_change():
    chunks = _overlapped(
        Block("Slide one.", "slide", {"slide_num": 1}),
        Block("Slide two.", "slide", {"slide_num": 2}),
    )
    assert chunks[1].text == "Slide two."


def test_overlap_within_one_oversized_slide_is_kept():
    long_slide = " ".join(f"Sentence number {i} here." for i in range(300))
    chunks = chunk_blocks(
        [Block(long_slide, "slide", {"slide_num": 1})], target_tokens=200, overlap_tokens=20
    )
    assert len(chunks) > 1
    assert chunks[1].text.startswith("Sentence number")
    assert chunks[1].text[:20] != chunks[0].text[:20]
    assert chunks[0].text.endswith(chunks[1].text.split("\n\n")[0])


def test_overlap_does_not_cross_a_sheet_change():
    first = Block("a" * 2000, "sheet_row", {"sheet": "A", "row_start": 2, "row_end": 21})
    second = Block("b" * 2000, "sheet_row", {"sheet": "B", "row_start": 2, "row_end": 21})
    chunks = chunk_blocks([first, second], target_tokens=500, overlap_tokens=80)
    assert chunks[1].text == "b" * 2000


def test_overlap_within_a_sheet_widens_the_cited_rows():
    first = Block("r" * 2000, "sheet_row", {"sheet": "A", "row_start": 2, "row_end": 21})
    second = Block("s" * 2000, "sheet_row", {"sheet": "A", "row_start": 22, "row_end": 41})
    chunks = chunk_blocks([first, second], target_tokens=500, overlap_tokens=80)
    assert chunks[1].meta["row_range"] == [2, 41]


def test_overlap_does_not_change_the_previous_chunks_metadata():
    chunks = _overlapped(
        Block("One.", "paragraph", {"page": 1, "heading_path": "S"}),
        Block("Two.", "paragraph", {"page": 2, "heading_path": "S"}),
        Block("Three.", "paragraph", {"page": 3, "heading_path": "S"}),
    )
    assert [c.meta["pages"] for c in chunks] == [[1], [1, 2], [2, 3]]
