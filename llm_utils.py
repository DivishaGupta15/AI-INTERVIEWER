# ── llm_utils.py ─────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Literal, List
import os

import openai
from pydantic import BaseModel


# ------------------------------------------------------------------ #
# 1.  Tiny pydantic schema to keep messages tidy
# ------------------------------------------------------------------ #
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# ------------------------------------------------------------------ #
# 2.  Core wrapper
# ------------------------------------------------------------------ #
def _gpt_chat(
    messages: List[ChatMessage],
    *,
    model: str = "gpt-3.5-turbo",          # or gpt-4o-mini, gpt-4o etc.
    temperature: float = 0.7,
) -> str:
    """
    Send a chat-completion request to OpenAI and return the **full**
    assistant reply as a plain string (no streaming).
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY env-var missing")

    resp = openai.chat.completions.create(
        model=model,
        messages=[m.model_dump() for m in messages],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------ #
# 3.  Public helpers
# ------------------------------------------------------------------ #
def summarise_resume(resume_text: str) -> str:
    messages = [
        ChatMessage(role="system", content="You are a recruitment assistant."),
        ChatMessage(
            role="user",
            content=(
                "Summarise the following résumé in exactly THREE concise "
                "bullet-lines, each on its own line. Focus on degree, key tech "
                "skills, major projects and leadership. Keep each line ≤ 20 "
                "words.\n\n"
                f"{resume_text}"
            ),
        ),
    ]
    return _gpt_chat(messages)


def summarise_jd(jd_text: str) -> str:
    messages = [
        ChatMessage(role="system",
                    content="You summarise job-descriptions for recruiters."),
        ChatMessage(
            role="user",
            content=(
                "Summarise the JD below in MAX FOUR short bullet-lines. "
                "Include role core-focus, mandatory tech / experience, any "
                "nice-to-haves, and key perks.\n\n"
                f"{jd_text}"
            ),
        ),
    ]
    return _gpt_chat(messages)


def rank_skills(skills: set[str], resume: str) -> list[str]:
    prompt = (
        "Below is a candidate résumé followed by a list of skills.\n"
        "Pick the FIVE most important skills for this candidate. "
        "Reply with ONLY those skills, comma-separated on a single line.\n\n"
        f"RESUME:\n{resume}\n\n"
        f"SKILLS:\n{', '.join(sorted(skills))}"
    )
    raw = _gpt_chat([ChatMessage(role="user", content=prompt)], temperature=0)
    first_line = raw.splitlines()[0]
    return [s.strip() for s in first_line.split(",") if s.strip()][:5]