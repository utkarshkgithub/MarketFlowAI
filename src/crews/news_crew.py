import json
import logging
from uuid import uuid4

from src.schemas.evidence import Evidence
from src.tools.news_fetcher import fetch_news
from src.tools.signal_calculators import check_red_flags
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior financial news analyst. 
You will be given raw news headlines for a stock ticker.
Your job is to analyze each headline and return a JSON array of evidence objects.
Each object must have exactly these fields:
  - claim: string (one clear, testable statement derived from the headline)
  - confidence: float between 0.0 and 1.0
  - sentiment: string ("positive", "negative", or "neutral")

Return ONLY a valid JSON array. No markdown, no explanation, no extra text."""


class NewsCrew:
    def execute(self, inputs: dict) -> list[Evidence]:
        ticker = inputs.get("ticker", "UNKNOWN")
        news_items = fetch_news(ticker)
        red_flags = check_red_flags(news_items)

        if not news_items:
            logger.warning("NewsCrew: No news items found for %s", ticker)
            return []

        headlines_text = "\n".join(
            f"- {item['title']} (date: {item.get('date', 'unknown')})"
            for item in news_items
        )

        user_prompt = (
            f"Analyze the following news headlines for the stock ticker {ticker}:\n\n"
            f"{headlines_text}\n\n"
            f"Also note these red flags were detected: {red_flags if red_flags else 'none'}.\n\n"
            f"Return a JSON array of evidence objects as instructed."
        )

        try:
            raw = chat(SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("NewsCrew LLM call failed: %s", e)
            return [
                Evidence(
                    id=str(uuid4()),
                    source_type="news",
                    source_ref="mock_news_api_fallback",
                    claim=f"Could not retrieve AI analysis for {ticker} news. Raw articles fetched: {len(news_items)}.",
                    confidence=0.3,
                    tags=["news", "fallback"],
                )
            ]

        evidences = []
        for item in parsed:
            sentiment = item.get("sentiment", "neutral")
            tags = ["news", "sentiment", sentiment]
            if red_flags:
                tags.append("risk")

            evidences.append(
                Evidence(
                    id=str(uuid4()),
                    source_type="news",
                    source_ref="news_api",
                    claim=item.get("claim", "No claim extracted."),
                    confidence=float(item.get("confidence", 0.7)),
                    tags=tags,
                )
            )

        return evidences
