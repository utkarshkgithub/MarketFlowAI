import json
import logging
from uuid import uuid4

from src.schemas.evidence import Evidence
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a neutral investment debate moderator.
You will be given a stock ticker and a summary of both bullish and bearish signals.
Your job is to adjudicate: which side has the stronger case, and why?
Return a JSON array with exactly 1 evidence object summarizing the outcome of the debate.
The object must have exactly these fields:
  - claim: string (one clear statement about which case won and the key deciding factor)
  - confidence: float between 0.0 and 1.0
  - tags: list of strings (e.g. ["debate", "verdict"])

Return ONLY a valid JSON array. No markdown, no explanation, no extra text."""


class DebateCrew:
    def execute(self, inputs: dict) -> list[Evidence]:
        ticker = inputs.get("ticker", "UNKNOWN")
        bull_signals = inputs.get("bull_signals", "Strong revenue growth, positive momentum")
        bear_signals = inputs.get("bear_signals", "High volatility, regulatory risk")

        user_prompt = (
            f"Adjudicate a bull vs bear debate for the stock {ticker}.\n\n"
            f"BULL CASE: {bull_signals}\n"
            f"BEAR CASE: {bear_signals}\n\n"
            f"Which side has the stronger case given these signals? "
            f"Return exactly 1 evidence object as a JSON array."
        )

        try:
            raw = chat(SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("DebateCrew LLM call failed: %s", e)
            return [
                Evidence(
                    id=str(uuid4()),
                    source_type="synthesis",
                    source_ref="debate_session_fallback",
                    claim=f"Could not complete AI debate adjudication for {ticker}.",
                    confidence=0.3,
                    tags=["debate", "fallback"],
                )
            ]

        evidences = []
        for item in parsed:
            evidences.append(
                Evidence(
                    id=str(uuid4()),
                    source_type="synthesis",
                    source_ref="debate_session",
                    claim=item.get("claim", "No debate outcome extracted."),
                    confidence=float(item.get("confidence", 0.6)),
                    tags=item.get("tags", ["debate", "verdict"]),
                )
            )

        return evidences
