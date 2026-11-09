"""Runs renderMarkdown from static/js/chat.js in Node, skipped if Node isn't installed."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_JS = Path(__file__).resolve().parent.parent / "static" / "js" / "chat.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node isn't installed")

_RUNNER = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const fn = (name) => {
  const start = src.indexOf("function " + name + "(");
  let depth = 0;
  for (let i = src.indexOf("{", start); ; i++) {
    if (src[i] === "{") depth++;
    if (src[i] === "}" && --depth === 0) return src.slice(start, i + 1);
  }
};
const input = JSON.parse(process.argv[2]);
const render = new Function(
  "currentSources",
  fn("escapeHtml") + ";" + fn("renderMarkdown") + ";return renderMarkdown;"
)(input.sources.map((n) => ({ n })));
console.log(JSON.stringify(render(input.text)));
"""


def _render(text: str, sources: tuple[int, ...] = (1, 2, 3)) -> str:
    result = subprocess.run(
        ["node", "-e", _RUNNER, "--", str(CHAT_JS), json.dumps({"text": text, "sources": sources})],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _chip(n: int) -> str:
    return f'<span class="citation" data-n="{n}">{n}</span>'


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Limit [1].", f"Limit {_chip(1)}."),
        ("Limit [1][2].", f"Limit {_chip(1)}{_chip(2)}."),
        ("Limit [1, 2].", f"Limit {_chip(1)}{_chip(2)}."),
        ("Limit [1-3].", f"Limit {_chip(1)}{_chip(2)}{_chip(3)}."),
        ("Limit [1–2].", f"Limit {_chip(1)}{_chip(2)}."),
        ("Limit [1-999].", f"Limit {_chip(1)}{_chip(2)}{_chip(3)}."),
    ],
)
def test_citations_for_known_sources_become_chips(text, expected):
    assert _render(text) == expected


@pytest.mark.parametrize("text", ["See [12].", "See [0].", "See [7-9]."])
def test_numbers_with_no_matching_source_stay_as_plain_text(text):
    assert _render(text) == text


def test_known_numbers_in_a_partly_invented_list_become_chips():
    assert _render("See [1, 12].") == f"See {_chip(1)}."


def test_inline_code_is_left_alone():
    assert _render("Use `arr[1]` here.") == "Use <code>arr[1]</code> here."


def test_fenced_code_is_left_alone():
    assert _render("```\nx = rows[1]\ny = **z**\n```") == (
        "<pre><code>x = rows[1]\ny = **z**</code></pre>"
    )


def test_text_around_code_is_still_formatted():
    assert _render("**Bold** `c` [2]\nnext") == (
        f"<strong>Bold</strong> <code>c</code> {_chip(2)}<br>next"
    )


def test_html_in_answers_and_code_is_escaped():
    assert _render("<img src=x> `<b>`") == "&lt;img src=x&gt; <code>&lt;b&gt;</code>"


def test_placeholder_characters_in_answers_cannot_inject_code_spans():
    assert _render("`a` \x000\x00") == "<code>a</code> 0"
