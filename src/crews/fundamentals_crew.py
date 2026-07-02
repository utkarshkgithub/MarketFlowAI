import json
import logging
from uuid import uuid4

from src.schemas.evidence import Evidence
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fundamental equity research analyst.
You will be given a stock ticker. Assume realistic public knowledge about the company's
financials (P/E ratio, revenue growth, debt levels, competitive moat).
Return a JSON array of 2-3 evidence objects analyzing the company's fundamentals.
Each object must have exactly these fields:
  - claim: string (one clear, testable statement about a fundamental metric)
  - confidence: float between 0.0 and 1.0
  - tags: list of strings (e.g. ["valuation", "fundamental"] or ["growth", "revenue"])

Return ONLY a valid JSON array. No markdown, no explanation, no extra text."""


class FundamentalsCrew:
    def execute(self, inputs: dict) -> list[Evidence]:
        ticker = inputs.get("ticker", "UNKNOWN")

        user_prompt = (
            f"Analyze the fundamental financial health of {ticker}. "
            f"Consider typical public metrics like P/E ratio, revenue growth trend, "
            f"debt-to-equity ratio, and competitive position. "
            f"Return 2-3 evidence objects as a JSON array."
        )

        try:
            raw = chat(SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("FundamentalsCrew LLM call failed: %s", e)
            return [
                Evidence(
                    id=str(uuid4()),
                    source_type="analysis",
                    source_ref="10-K_fallback",
                    claim=f"Could not retrieve AI fundamental analysis for {ticker}.",
                    confidence=0.3,
                    tags=["fundamental", "fallback"],
                )
            ]

        evidences = []
        for item in parsed:
            evidences.append(
                Evidence(
                    id=str(uuid4()),
                    source_type="analysis",
                    source_ref="fundamental_analysis",
                    claim=item.get("claim", "No claim extracted."),
                    confidence=float(item.get("confidence", 0.8)),
                    tags=item.get("tags", ["fundamental"]),
                )
            )

        return evidences
