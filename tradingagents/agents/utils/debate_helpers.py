"""Pure helpers for the simultaneous-debate transcript.

Leaf module (no imports of ``tradingagents.graph`` or the agent package) so
advocate agents can import it without the circular import that would arise
from importing ``tradingagents.graph.debate`` (the graph package imports the
agent package). ``graph/debate.py`` re-imports these for its conditional.
"""

from __future__ import annotations


def turns_for_side(turns: list[dict], side: str) -> list[dict]:
    """All turns logged by ``side``, in insertion order."""
    return [t for t in turns if t.get("side") == side]


def next_round_for_side(turns: list[dict], side: str) -> int:
    """Round an advocate of ``side`` should write next = its completed turns + 1."""
    return len(turns_for_side(turns, side)) + 1


def render_transcript(turns: list[dict], sides: tuple[str, ...]) -> str:
    """Render the transcript as a compact canonical side-by-side block.

    Rows = rounds (sorted), columns = sides in the fixed order given. This is
    what advocates read next round (opponent's prior deltas to rebut) AND what
    the synthesizer adjudicates — no speaking order, no narrative re-paste.
    """
    if not turns:
        return "(no turns yet)"
    by_round: dict[int, dict[str, str]] = {}
    for t in turns:
        by_round.setdefault(t.get("round"), {})[t.get("side")] = t.get("delta", "")
    lines = []
    for r in sorted(by_round):
        row = by_round[r]
        for s in sides:
            lines.append(f"R{r} {s}: {row.get(s, '(no turn)')}")
    return "\n".join(lines)
