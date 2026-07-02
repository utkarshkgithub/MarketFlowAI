import json
import logging
from uuid import uuid4

from src.schemas.evidence import Evidence
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an options market analyst specializing in liquidity and derivatives.
You will be given a stock ticker that has shown unusual volatility.
Analyze what a high-volatility environment means for this stock's options market.
Return a JSON array of 2 evidence objects.
Each object must have exactly these fields:
  - claim: string (one clear, testable statement about options flow or liquidity)
  - confidence: float between 0.0 and 1.0
  - tags: list of strings (e.g. ["options", "liquidity"] or ["put_call", "bearish"])

Return ONLY a valid JSON array. No markdown, no explanation, no extra text."""


class OptionsLiquidityCrew:
    def execute(self, inputs: dict) -> list[Evidence]:
        ticker = inputs.get("ticker", "UNKNOWN")

        user_prompt = (
            f"The stock {ticker} has shown a significant volatility spike. "
            f"Analyze what this typically means for the options market: "
            f"consider put/call ratio, implied volatility levels, and liquidity risk. "
            f"Return 2 evidence objects as a JSON array."
        )

        try:
            raw = chat(SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("OptionsLiquidityCrew LLM call failed: %s", e)
            return [
                Evidence(
                    id=str(uuid4()),
                    source_type="market_data",
                    source_ref="options_chain_fallback",
                    claim=f"Could not retrieve AI options analysis for {ticker}.",
                    confidence=0.3,
                    tags=["options", "fallback"],
                )
            ]

        evidences = []
        for item in parsed:
            evidences.append(
                Evidence(
                    id=str(uuid4()),
                    source_type="market_data",
                    source_ref="options_chain",
                    claim=item.get("claim", "No claim extracted."),
                    confidence=float(item.get("confidence", 0.75)),
                    tags=item.get("tags", ["options"]),
                )
            )

        return evidences
