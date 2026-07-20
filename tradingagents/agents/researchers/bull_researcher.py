"""Bull Researcher — simultaneous-debate advocate for the long case.

Round-aware: reads the canonical transcript, infers its round from its own
turn count, and writes ONLY a delta (opening position in round 1; new evidence
+ named rebuttals in later rounds). See graph/debate.py for the topology.
"""

from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    collect_analyst_reports,
    format_fair_value_block,
    get_instrument_context_from_state,
    get_language_instruction,
    safe_llm_invoke,
)
from tradingagents.agents.utils.debate_helpers import next_round_for_side, render_transcript

_SIDE = "bull"
_OPPONENT = "bear"
_SIDES = ("bull", "bear")


_BULL_SYSTEM = """ROLE
You are the Bull Researcher — a partisan advocate for the LONG case in a structured adversarial debate. Your opponent is the Bear Researcher.

ABSOLUTE OUTPUT CONTRACT (every response must obey these):
1. Your output ALWAYS begins with the literal line `**Round {my_round}**` on its own line. This is non-negotiable and identifies which round the supervisor records.
2. Output ONLY the body for THIS round. Do NOT emit summary / verdict / probability blend / stop-loss / position-sizing — those belong to the Research Manager, NOT you.
3. Round 1 — OPENING (no opponent turn exists yet). Write:
   - One sentence: core bull thesis.
   - 3-5 cited evidence points (name the source analyst + the number for each).
   - A short "Acknowledged risks" line (1-2 items, NOT arguments against yourself).
4. Round 2+ — DELTA-ONLY (opponent's turn NOW EXISTS in the data block). You MUST:
   - Read the opponent's last delta in the TRANSCRIPT and quote a specific claim from it.
   - Write only (a) a named rebuttal to that specific claim, or (b) new evidence the opponent has not yet seen.
   - Cite the opponent's line as `Re: R{my_round-1}-{opp}` (e.g. `Re: R1-bear`).
   - Do NOT restate, paraphrase, or repeat your prior round. Do NOT write opening-style content. Do NOT write a verdict.

HARD CONSTRAINTS
- Cite source analyst + number for every claim.
- Disagree on interpretation, never on facts.
- If opponent has no turn yet (transcript shows `<no turn>` for them), treat Round 1 as opening and do NOT fabricate opponent claims to rebut."""


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        turns = state.get("research_debate_turns", [])
        my_round = next_round_for_side(turns, _SIDE)
        transcript = render_transcript(turns, _SIDES)
        report_digest = state.get("report_digest") or collect_analyst_reports(state)
        instrument_context = get_instrument_context_from_state(state)
        fv_context = format_fair_value_block(state)

        # Per-round instruction makes "this is R1 vs R2" explicit to the model.
        round_directive = (
            "Round 1 (OPENING). No opponent turn exists yet. Write the opening position following the OUTPUT CONTRACT; "
            "no verdict, no probability blend, no stop-loss."
            if my_round == 1
            else
            f"Round {my_round} (DELTA-ONLY). Your opponent's turn IS in the transcript. "
            f"You MUST (a) pick a specific claim from `Re: R{my_round-1}-{_OPPONENT}` and rebut it, "
            f"or (b) introduce new evidence the opponent has not yet seen. "
            f"No restatement of your R1. No verdict. No probability blend. No stop-loss."
        )

        data_block = (
            f"{fv_context}\n\n"
            f"ANALYST REPORT DIGEST:\n{report_digest}\n\n"
            f"DEBATE TRANSCRIPT (canonical; opponent = {_OPPONENT}):\n{transcript}\n\n"
            f"YOUR ROUND: {my_round}\n"
            f"INSTRUCTION FOR THIS ROUND: {round_directive}\n\n"
            + get_language_instruction()
        )

        system_message = _BULL_SYSTEM.replace("{my_round}", str(my_round))

        prompt_template = chat_prompt_messages(
            system_message, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = safe_llm_invoke(llm, formatted, "Bull Researcher")

        return {"research_debate_turns": [{"round": my_round, "side": _SIDE, "delta": response.content}]}

    return bull_node
