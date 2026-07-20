"""Simultaneous-debate primitive — order-neutral adversarial exchange.

Both advocates fan out IN PARALLEL each round, appending a turn dict
``{"round", "side", "delta"}`` to a shared list on state. The list uses an
``operator.add`` reducer, so the two concurrent appends merge without the
nested-TypedDict collision that forced the old separate-key workaround.

A no-op **dispatcher** node fans out each round; a no-op **collector** node
fans in and hangs the conditional. After ``max_rounds`` rounds the conditional
routes to the synthesizer; otherwise back to the dispatcher.

Round count is DERIVED from the transcript: each advocate infers its next
round from its own turn count (``next_round_for_side``), so the primitive
needs no external counter on state.
"""

import logging
from typing import Callable

from langgraph.graph import StateGraph

from tradingagents.agents.utils.debate_helpers import turns_for_side

logger = logging.getLogger(__name__)


def _noop(_state) -> dict:
    """No-op node: fan-out dispatcher / fan-in collector."""
    return {}


def make_should_continue_debate(
    turns_key: str, sides: tuple[str, ...], max_rounds: int
) -> Callable[[dict], str]:
    """Conditional: both sides have logged ``max_rounds`` turns → 'synthesizer', else 'advocates'."""

    def _should_continue(state: dict) -> str:
        turns = state.get(turns_key, [])
        per_side = {s: len(turns_for_side(turns, s)) for s in sides}
        done = all(v >= max_rounds for v in per_side.values())
        decision = "synthesizer" if done else "advocates"
        logger.info("debate router [%s]: per_side=%s max=%s -> %s",
                    turns_key, per_side, max_rounds, decision)
        return decision

    return _should_continue


def build_debate(
    workflow: StateGraph,
    *,
    entry_from: str,
    advocates: dict[str, str],  # {node_name: side}, e.g. {"Bull Researcher": "bull"}
    synthesizer: str,
    turns_key: str,
    max_rounds: int,
) -> None:
    """Wire a simultaneous-rounds debate into ``workflow``.

    Topology::

        entry_from → dispatcher → [advocate_a, advocate_b]   (fan-out, parallel)
        advocate_a, advocate_b → collector                   (fan-in)
        collector --(rounds done?)--> dispatcher | synthesizer

    Advocates must already be added as nodes. ``dispatcher`` (fan-out) and
    ``collector`` (fan-in) no-op nodes are added by this helper. Advocate nodes
    must read ``state[turns_key]`` to infer their round and write their turn
    delta back to the same key.
    """
    sides = tuple(advocates.values())
    advocate_nodes = list(advocates.keys())
    dispatcher = f"{entry_from}__debate_dispatch"
    collector = f"{entry_from}__debate_collect"

    workflow.add_node(dispatcher, _noop)
    workflow.add_node(collector, _noop)

    # entry → dispatcher → both advocates (parallel fan-out)
    workflow.add_edge(entry_from, dispatcher)
    for node in advocate_nodes:
        workflow.add_edge(dispatcher, node)

    # both advocates → collector (fan-in)
    for node in advocate_nodes:
        workflow.add_edge(node, collector)

    # collector → next round (dispatcher) OR synthesizer
    workflow.add_conditional_edges(
        collector,
        make_should_continue_debate(turns_key, sides, max_rounds),
        {"advocates": dispatcher, "synthesizer": synthesizer},
    )
