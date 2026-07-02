import json
import logging
from uuid import uuid4

from src.schemas.evidence import Evidence
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a legal and regulatory risk analyst covering public companies.
You will be given a stock ticker and any news red flags detected.
Your job is to assess the legal and regulatory risk environment for this company.
Return a JSON array of 1-2 evidence objects.
Each object must have exactly these fields:
  - claim: string (one clear, testable statement about legal or regulatory risk)
  - confidence: float between 0.0 and 1.0
  - tags: list of strings (e.g. ["legal", "compliance"] or ["regulatory", "sec"])

Return ONLY a valid JSON array. No markdown, no explanation, no extra text."""


class RegulationLegalCrew:
    def execute(self, inputs: dict) -> list[Evidence]:
        ticker = inputs.get("ticker", "UNKNOWN")
        issues = inputs.get("issues", [])

        user_prompt = (
            f"Assess the legal and regulatory risk for {ticker}. "
            f"The following red flags were detected in recent news: {issues if issues else 'none specifically identified'}. "
            f"Consider known litigation history, SEC investigations, and regulatory exposure. "
            f"Return 1-2 evidence objects as a JSON array."
        )

        try:
            raw = chat(SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("RegulationLegalCrew LLM call failed: %s", e)
            return [
                Evidence(
                    id=str(uuid4()),
                    source_type="legal_db",
                    source_ref="court_filings_fallback",
                    claim=f"Could not retrieve AI legal analysis for {ticker}.",
                    confidence=0.3,
                    tags=["legal", "fallback"],
                )
            ]

        evidences = []
        for item in parsed:
            evidences.append(
                Evidence(
                    id=str(uuid4()),
                    source_type="legal_db",
                    source_ref="regulatory_filings",
                    claim=item.get("claim", "No claim extracted."),
                    confidence=float(item.get("confidence", 0.85)),
                    tags=item.get("tags", ["legal"]),
                )
            )

        return evidences
