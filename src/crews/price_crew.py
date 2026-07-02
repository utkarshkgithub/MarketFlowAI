import json
import logging
from uuid import uuid4

from src.schemas.evidence import Evidence
from src.tools.price_fetcher import fetch_prices
from src.tools.signal_calculators import compute_volatility, compute_drawdown
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative financial analyst.
You will be given raw price statistics for a stock (volatility and max drawdown).
Your job is to interpret these numbers and return a JSON array of evidence objects.
Each object must have exactly these fields:
  - claim: string (one clear, testable statement interpreting the statistic)
  - confidence: float between 0.0 and 1.0
  - tags: list of strings (e.g. ["volatility", "risk"] or ["drawdown", "momentum"])

Return ONLY a valid JSON array. No markdown, no explanation, no extra text."""


class PriceCrew:
    def execute(self, inputs: dict) -> list[Evidence]:
        ticker = inputs.get("ticker", "UNKNOWN")
        prices = fetch_prices(ticker)

        volatility = compute_volatility(prices)
        drawdown = compute_drawdown(prices)

        user_prompt = (
            f"Analyze the following price statistics for {ticker}:\n\n"
            f"- 20-day Volatility: {volatility:.2%}\n"
            f"- Max Drawdown (20d): {drawdown:.2%}\n\n"
            f"Interpret what these numbers mean for an investor and return 2 evidence objects "
            f"(one for volatility, one for drawdown) as a JSON array."
        )

        try:
            raw = chat(SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("PriceCrew LLM call failed: %s", e)
            return [
                Evidence(
                    id=str(uuid4()),
                    source_type="price",
                    source_ref="mock_price_feed",
                    claim=f"Volatility for {ticker} is {volatility:.2%}",
                    confidence=0.95,
                    raw_snippet=str(prices[-5:]),
                    tags=["volatility", "risk"],
                ),
                Evidence(
                    id=str(uuid4()),
                    source_type="price",
                    source_ref="mock_price_feed",
                    claim=f"Max drawdown for {ticker} is {drawdown:.2%}",
                    confidence=1.0,
                    tags=["drawdown", "risk"],
                ),
            ]

        evidences = []
        for item in parsed:
            evidences.append(
                Evidence(
                    id=str(uuid4()),
                    source_type="price",
                    source_ref="price_feed",
                    claim=item.get("claim", "No claim extracted."),
                    confidence=float(item.get("confidence", 0.9)),
                    raw_snippet=str(prices[-3:]),
                    tags=item.get("tags", ["price"]),
                )
            )

        return evidences
