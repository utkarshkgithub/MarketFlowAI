import json
import logging
from typing import List

from src.schemas.evidence import Evidence
from src.schemas.report import Signals
from src.crews.llm_client import chat

logger = logging.getLogger(__name__)

_RATIONALE_SYSTEM_PROMPT = """You are a senior equity research writer.
You will be given a list of evidence claims about a stock (both bullish and bearish).
Write a concise, professional investment rationale split into exactly two parts:
1. BULL CASE: 2-3 sentences summarizing the strongest bullish arguments.
2. BEAR CASE: 2-3 sentences summarizing the strongest bearish arguments.

Return ONLY valid JSON with exactly two keys: "bull_case" and "bear_case".
Each value must be a plain string (no markdown, no bullet points inside). No extra keys."""


class Synthesizer:
    def build_signals(self, evidences: List[Evidence]) -> Signals:
        # Base Raw Signals
        volatility = None
        drawdown = None
        red_flags = []
        
        # New Features Accumulators
        sentiment_sum = 0.0
        sentiment_count = 0
        
        momentum_sum = 0.0
        momentum_count = 0
        
        event_risk_score = 0.0
        uncertainty_accum = 0.0
        
        for ev in evidences:
            # 1. Parse Volatility/Drawdown (Price)
            if "volatility" in ev.tags and "Volatility for" in ev.claim:
                try:
                    val_str = ev.claim.split(" is ")[-1].replace("%", "")
                    volatility = float(val_str) / 100.0
                except:
                    pass
            
            if "drawdown" in ev.tags and "drawdown for" in ev.claim:
                    try:
                         val_str = ev.claim.split(" is ")[-1].replace("%", "")
                         drawdown = float(val_str) / 100.0
                    except:
                        pass
            
            # 2. News/Legal Red Flags
            if "legal" in ev.tags and "red flags" in ev.claim:
                # "Identified potential red flags: flag1, flag2"
                flags = ev.claim.split(": ")[-1].split(", ")
                red_flags.extend(flags)
                event_risk_score = max(event_risk_score, 0.9) # High risk if red flags
            
            # 3. Sentiment Extraction
            if "sentiment" in ev.tags or "news" in ev.source_type:
                # Look for keywords in claim
                if "record breaking" in ev.claim or "positive" in ev.claim or "Buy" in ev.claim:
                    sentiment_sum += 0.8
                    sentiment_count += 1
                elif "lawsuit" in ev.claim or "negative" in ev.claim:
                    sentiment_sum -= 0.8
                    sentiment_count += 1
                    
            # 4. Momentum/Valuation
            if "valuation" in ev.tags:
                if "undervalued" in ev.claim:
                    momentum_sum += 0.5
                    momentum_count += 1
                elif "overvalued" in ev.claim:
                    momentum_sum -= 0.5
                    momentum_count += 1
                    
            # 5. Uncertainty
            # If evidence confidence is low, uncertainty is high
            uncertainty_accum += (1.0 - ev.confidence)

        # Compute Final Feature Scores
        final_sentiment = sentiment_sum / sentiment_count if sentiment_count > 0 else 0.0
        final_momentum = momentum_sum / momentum_count if momentum_count > 0 else 0.0
        
        # Volatility Risk mapping
        vol_risk = 0.0
        if volatility:
            if volatility > 0.4: vol_risk = 0.9
            elif volatility > 0.2: vol_risk = 0.5
            else: vol_risk = 0.1
            
        # Uncertainty normalization
        avg_uncertainty = uncertainty_accum / len(evidences) if evidences else 1.0

        return Signals(
            volatility_20d=volatility,
            volatility_risk=vol_risk,
            drawdown_20d=drawdown,
            news_red_flags=red_flags,
            event_risk=event_risk_score,
            sentiment_score=final_sentiment,
            momentum_score=final_momentum,
            uncertainty=avg_uncertainty,
            conflict_score=0.1 
        )

    def generate_rationale(self, evidences: List[Evidence], ticker: str) -> dict:
        """
        Uses OpenAI to write a professional Bull/Bear narrative
        from the full list of collected Evidence objects.
        Falls back to a simple text summary if the LLM call fails.
        """
        claims_text = "\n".join(
            f"- [{ev.source_type}] {ev.claim} (confidence: {ev.confidence:.0%})"
            for ev in evidences
        )

        user_prompt = (
            f"Here are all the evidence claims gathered for the stock {ticker}:\n\n"
            f"{claims_text}\n\n"
            f"Write the BULL CASE and BEAR CASE rationale as instructed."
        )

        try:
            raw = chat(_RATIONALE_SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(raw)
            return {
                "bull_case": parsed.get("bull_case", "No bull case generated."),
                "bear_case": parsed.get("bear_case", "No bear case generated."),
            }
        except Exception as e:
            logger.error("Synthesizer.generate_rationale LLM call failed: %s", e)
            # Fallback: simple split of evidence into positive/negative
            bull = [ev.claim for ev in evidences if ev.confidence >= 0.8]
            bear = [ev.claim for ev in evidences if ev.confidence < 0.8]
            return {
                "bull_case": " ".join(bull) if bull else "No strong bull signals found.",
                "bear_case": " ".join(bear) if bear else "No strong bear signals found.",
            }
