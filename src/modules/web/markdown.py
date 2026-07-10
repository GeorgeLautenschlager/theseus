"""Minimal markdown -> HTML rendering for chat bubbles.

Supports the subset the Theseus Chat UI design calls for: **bold**, *italic*,
inline `code`, fenced ```code blocks```, and `-` bullet lists. Everything is
HTML-escaped before any markdown syntax is applied, so agent output can never
inject markup into the page.
"""

from __future__ import annotations

import html
import re

_FENCE_RE = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(^|[^*])\*([^*]+)\*(?!\*)")
_BULLET_RE = re.compile(r"^-\s+(.*)$")
_FENCE_PLACEHOLDER_RE = re.compile(r"@@CODEBLOCK(\d+)@@")


def render_markdown(text: str) -> str:
    if not text:
        return ""

    code_blocks: list[str] = []

    def _stash_fence(match: re.Match[str]) -> str:
        code_blocks.append(match.group(2).rstrip("\n"))
        return f"@@CODEBLOCK{len(code_blocks) - 1}@@"

    working = _FENCE_RE.sub(_stash_fence, text)
    working = html.escape(working)
    working = _INLINE_CODE_RE.sub(
        lambda m: f'<code class="inline-code">{m.group(1)}</code>', working
    )
    working = _BOLD_RE.sub(r"<strong>\1</strong>", working)
    working = _ITALIC_RE.sub(r"\1<em>\2</em>", working)

    out: list[str] = []
    in_list = False
    for line in working.split("\n"):
        stripped = line.strip()
        if _FENCE_PLACEHOLDER_RE.fullmatch(stripped):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(stripped)
            continue
        bullet = _BULLET_RE.match(stripped)
        if bullet:
            if not in_list:
                out.append('<ul class="bubble-list">')
                in_list = True
            out.append(f"<li>{bullet.group(1)}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("<br/>" if stripped == "" else f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")

    html_out = "\n".join(out)

    def _restore_fence(match: re.Match[str]) -> str:
        code = html.escape(code_blocks[int(match.group(1))])
        return f'<pre class="code-block"><code>{code}</code></pre>'

    return _FENCE_PLACEHOLDER_RE.sub(_restore_fence, html_out)
