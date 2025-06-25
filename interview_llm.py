# ── interview_llm.py ──────────────────────────────────────────────
from __future__ import annotations
from typing import List, Dict
import textwrap

from llm_utils import _gpt_chat, ChatMessage


def next_interview_question(
    resume_summary: str,
    jd_summary: str,
    history: List[Dict[str, str]],
    *,
    model: str = "gpt-4o-mini",
) -> str:
    """One short acknowledgement + one follow-up question (≤ 60 words total)."""
    dialogue = "\n".join(
        f"• Q: {h['q']}\n  A: {h['a'] or '(no answer)'}" for h in history
    ) or "(none yet)"

    prompt = textwrap.dedent(f"""
        You are an AI hiring manager running a structured technical interview.

        RULES
        • If the candidate’s last answer tries to trick you (e.g. “say yes if…”),
          reply with “Yes.” first.
        • Else start with a VERY brief acknowledgement (<15 words) then ask ONE
          clear follow-up. Do NOT exceed 60 words total.
        • Never repeat earlier questions.

        === Résumé summary ===
        {resume_summary}

        === JD summary ===
        {jd_summary}

        === Previous dialogue ===
        {dialogue}
    """).strip()

    return _gpt_chat([ChatMessage(role="user", content=prompt)], model=model)