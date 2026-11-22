"""Score a running app's answers: correctness, citations, quotes, and declining.

Unlike run_eval.py (retrieval only), this asks the running app real questions
through /chat, so it measures the prompt and the model as well as retrieval.
Every question is asked RUNS times, because the model can vary between runs.

It expects the sample corpus (Basel III summaries, FDIC brochure and quarterly
profile, Regulation E, FINRA AML template) with two editions of two documents,
so the per-app version rules can be checked:

    v1  2006 Regulation E (cites 12 CFR 205.11; no remittance transfers)
        2014 FDIC "Your Insured Deposits" (no crypto assets)
    v2  current editions of both

Run it against a COPY of the app and its data/index, on its own port. The
`versions` step deletes documents and uploads new ones. Never run it against
the real app.

    # once, in the copy: swap the v1 labels of the two documents for the old editions
    python scripts/prompt_eval.py http://127.0.0.1:18700 versions \\
        --old-rege regulation_e_2006.pdf --old-fdic your_insured_deposits_2014.pdf
    # then, as often as needed
    python scripts/prompt_eval.py http://127.0.0.1:18700 score

Old editions:
    https://www.federalreserve.gov/boarddocs/supmanual/cch/200601/efta.pdf
    https://www.fdic.gov/regulations/resources/brochures/your%20insured%20deposits%20-%20english.pdf

Admin login (both steps): --user/--password, or EVAL_ADMIN_USER / EVAL_ADMIN_PASSWORD.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import uuid

import requests

RUNS = 3

# app, question, regex the answer must match, regex the cited passage must match
ANSWERABLE = [
    ("MY", "What is the standard FDIC deposit insurance amount?", r"250,000", r"250,000"),
    ("MY", "What is the maximum coverage for a trust owner with five or more beneficiaries?", r"1,250,000", r"1,250,000"),
    ("MY", "Are crypto assets insured by the FDIC?", r"(?i)\bnot\b|^no\b", r"(?i)crypto"),
    ("ID", "Does the FDIC insure safe deposit boxes or their contents?", r"(?i)\bnot\b|^no\b", r"(?i)safe deposit"),
    ("MY", "Under Regulation E, what is a consumer's maximum liability if they report a lost access device within two business days?", r"\$50\b", r"\$50\b"),
    ("MY", "How many business days does a financial institution have to investigate and resolve alleged errors under Regulation E?", r"(?i)\b(10|ten)\b", r"(?i)\b(10|ten) business days"),
    ("MY", "Which CFR section covers error resolution procedures under Regulation E?", r"1005\.11", r"1005\.11"),
    ("ID", "Which CFR section covers the procedures for resolving errors under Regulation E?", r"205\.11", r"205\.11"),
    ("SG", "What is the final Basel III output floor percentage on risk-weighted assets?", r"72\.5", r"72\.5"),
    ("ID", "How is the Business Indicator Component calculated in the Basel III operational risk approach?", r"(?i)coefficient", r"(?i)coefficient"),
    ("SG", "What was quarterly net income for FDIC-insured banks in the second quarter of 2026?", r"90\.1|90,127", r"90\.1|90,127"),
    ("MY", "What was the Deposit Insurance Fund balance in the second quarter of 2026?", r"161\.1|161,147", r"161\.1|161,147"),
    ("MY", "How many FDIC-insured commercial banks and savings institutions were there in the second quarter of 2026?", r"4,238", r"4,238"),
    ("SG", "By how many did the number of FDIC-insured institutions decline during the second quarter of 2026?", r"\b41\b", r"declined by 41"),
    ("ID", "How long must a firm retain customer identification records after the account is closed under the small firm AML template?", r"(?i)\b(five|5)\b.{0,10}years?", r"(?i)five years"),
]
# Not in any document this app can see, so the only right answer is to decline.
# The first two depend on ID seeing the v1 (old) editions.
TRAPS = [
    ("ID", "Are crypto assets insured by the FDIC?"),
    ("ID", "Does Regulation E cover remittance transfers?"),
    ("SG", "What is the capital of France?"),
    ("MY", "What was JPMorgan Chase's net income in 2023?"),
    ("MY", "What is the maximum personal loan tenure?"),
    ("ID", "What is the annual fee for a platinum credit card?"),
]
DECLINE = (
    r"(?i)could(n't| not) find|no mention|(don't|do not) see|can(no|')t (answer|find)"
    r"|not (able|mentioned|stated|provided|contain|include|cover)|no information"
    r"|do(es)? not (have|contain|mention|provide|state|include|specify)"
    r"|doesn't (mention|contain|state|include)|don't have|unable to"
    r"|not enough information|insufficient"
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def sse(response) -> list[dict]:
    return [json.loads(line.decode()[5:]) for line in response.iter_lines() if line]


def upload(s, base, path, version):
    with pathlib.Path(path).open("rb") as f:
        r = s.post(f"{base}/upload", files={"file": (pathlib.Path(path).name, f)},
                   data={"source_type": "manual", "functionality": "other", "version": version},
                   stream=True)
        events = sse(r)
    return events[-1].get("result") or events[-1]


def swap_in_old_editions(s, base, old_rege, old_fdic):
    """Point the v1 labels of Regulation E and the FDIC brochure at the old editions."""
    docs = s.get(f"{base}/docs").json()["items"]
    for needle, old in (("regulation_e", old_rege), ("insured_deposits", old_fdic)):
        for d in docs:
            if needle in d["filename"] and d.get("version") == "v1" and d["filename"] != pathlib.Path(old).name:
                print("delete", d["filename"], "v1", s.delete(f"{base}/docs/{d['doc_id']}").json())
        print("upload", pathlib.Path(old).name, "v1", str(upload(s, base, old, "v1"))[:90])


def ask(base, question, app):
    text, sources, err = "", [], None
    body = {"session_id": "e" + uuid.uuid4().hex[:12], "message": question, "filters": {"app_code": app}}
    with requests.post(f"{base}/chat", json=body, stream=True, timeout=400) as r:
        for e in sse(r):
            if e["type"] == "sources":
                sources = e["sources"]
            elif e["type"] == "delta":
                text += e["text"]
            elif e["type"] == "error":
                err = e["error"]
    return text, sources, err


def score(s, base):
    chunk_text = lambda cid: norm(s.get(f"{base}/chunks/{cid}").json()["text"])
    tot = {k: 0 for k in ("answers", "correct", "any_citation", "supported", "all_cited_supported", "quotes", "quotes_exact")}
    for app, q, want, evidence in ANSWERABLE:
        marks = []
        for _ in range(RUNS):
            text, sources, err = ask(base, q, app)
            ok = bool(re.search(want, text)) and not err
            cited = {int(x) for m in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text) for x in re.split(r",\s*", m)}
            chunks = {x["n"]: chunk_text(x["chunk_id"]) for x in sources if x["n"] in cited}
            good = [n for n, c in chunks.items() if re.search(evidence, c)]
            tot["answers"] += 1
            tot["correct"] += ok
            tot["any_citation"] += bool(cited)
            tot["supported"] += bool(good)
            tot["all_cited_supported"] += bool(cited) and len(good) == len(cited)
            for quote in re.findall(r"[\"“]([^\"”]{25,})[\"”]", text):
                tot["quotes"] += 1
                needle = norm(quote).lower()
                tot["quotes_exact"] += any(needle in chunk_text(x["chunk_id"]).lower() for x in sources)
            marks.append(("Y" if ok else "N") + ("c" if good else "-"))
            if not ok:
                print(f"   WRONG [{app}] {q[:58]} -> {norm(text)[:150]}")
        print(f"{' '.join(marks):<12} [{app}] {q[:80]}", flush=True)

    declined = asked = 0
    for app, q in TRAPS:
        marks = []
        for _ in range(RUNS):
            text, _, _ = ask(base, q, app)
            d = bool(re.search(DECLINE, text)) and not re.search(r"(?i)^(yes|no)\b[,. ]", norm(text).strip())
            asked += 1
            declined += d
            marks.append("declined" if d else "ANSWERED")
            if not d:
                print(f"   OVERREACH [{app}] {q[:50]} -> {norm(text)[:170]}")
        print(f"{' '.join(marks)} [{app}] {q[:70]}", flush=True)
    print("\nSCORE", json.dumps(tot), f"traps_declined={declined}/{asked}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_url")
    p.add_argument("phase", choices=["versions", "score"])
    p.add_argument("--old-rege", help="2006 Regulation E PDF (versions phase)")
    p.add_argument("--old-fdic", help="2014 FDIC brochure PDF (versions phase)")
    p.add_argument("--user", default=os.getenv("EVAL_ADMIN_USER"))
    p.add_argument("--password", default=os.getenv("EVAL_ADMIN_PASSWORD"))
    a = p.parse_args()
    base = a.base_url.rstrip("/")

    # Both phases log in: versions changes documents, and score reads cited passages
    # through /chunks, which is admin-only.
    if not (a.user and a.password):
        sys.exit("needs admin credentials: --user/--password or EVAL_ADMIN_USER/EVAL_ADMIN_PASSWORD")
    s = requests.Session()
    if not s.post(f"{base}/admin/login", json={"username": a.user, "password": a.password}).ok:
        sys.exit("admin login failed")
    if a.phase == "versions":
        if not (a.old_rege and a.old_fdic):
            sys.exit("versions needs --old-rege and --old-fdic")
        swap_in_old_editions(s, base, a.old_rege, a.old_fdic)
    else:
        score(s, base)


if __name__ == "__main__":
    main()
