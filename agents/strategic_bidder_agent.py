# agents/strategic_bidder_agent.py

from __future__ import annotations

from core.models import BidRequest, BidResponse
from .base_agent import BaseAgent
from tools.best_response import approximate_best_response
import json


class StrategicBidderAgent(BaseAgent):
    """
    LLM-backed strategic bidder.

    - Calls approximate_best_response() tool to get a recommended best bid.
    - Passes that recommendation into the LLM prompt.
    - LLM returns a JSON bid + reasoning.
    """

    def __init__(self, bidder_id: str, use_llm: bool = True):
        super().__init__(name=f"StrategicBidder-{bidder_id}")
        self.bidder_id = bidder_id
        self.use_llm = use_llm

        self._shared = self.load_prompt_template("shared_instructions.txt")
        self._persona = self.load_prompt_template("strategic_bidder_prompt.txt")

    def build_prompt(self, request: BidRequest, recommended_bid: float, expected_utility: float) -> str:
        if request.history:
            history_text = json.dumps(request.history, indent=2)
        else:
            history_text = "No history available."

        return f"""
{self._shared}

{self._persona}

Information:
- bidder_id: "{self.bidder_id}"
- private_value: {request.private_value}
- best_response_calculator_recommendation:
    - best_bid: {recommended_bid}
    - expected_utility: {expected_utility}

history: "{history_text}"
""".strip()

    def get_bid(self, request: BidRequest) -> BidResponse:
        # Tool-based best response (pure Python)
        result = approximate_best_response(private_value=request.private_value)
        recommended_bid = float(result["best_bid"])
        expected_utility = float(result["expected_utility"])

        if not self.use_llm:
            auction_id = request.auction_id
            round_index = request.round_index
            # No LLM: just use the tool's suggested best bid
            return BidResponse(
                auction_id=auction_id,
                round_index=round_index,
                bidder_id=self.bidder_id,
                bid=recommended_bid,
                reasoning="Used approximate_best_response tool directly (no LLM).",
                raw_llm_text=None,
            )

        prompt = self.build_prompt(
            request=request,
            recommended_bid=recommended_bid,
            expected_utility=expected_utility,
        )

        try:
            raw_text = self.call_llm(prompt)
        except NotImplementedError:
            auction_id = request.auction_id
            round_index = request.round_index
            # If LLM is not wired, fall back to tool recommendation
            return BidResponse(
                auction_id=auction_id,
                round_index=round_index,
                bidder_id=self.bidder_id,
                bid=recommended_bid,
                reasoning="LLM not configured, used approximate_best_response tool only.",
                raw_llm_text=None,
            )

        return self.parse_bid_response(
            request,
            raw_text
        )
