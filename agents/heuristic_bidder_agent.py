# agents/heuristic_bidder_agent.py

from __future__ import annotations

from typing import Optional

from core.models import BidRequest, BidResponse
from .base_agent import BaseAgent
import json


class HeuristicBidderAgent(BaseAgent):
    """
    LLM-backed heuristic bidder.

    - Uses a simple shading rule as a fallback.
    - Optionally calls an LLM to wrap that rule in "reasoning".
    """

    def __init__(self, bidder_id: str, shading_factor: float = 0.8, use_llm: bool = True):
        super().__init__(name=f"HeuristicBidder-{bidder_id}")
        self.bidder_id = bidder_id
        self.shading_factor = shading_factor
        self.use_llm = use_llm

        # Pre-load templates if you want
        self._shared = self.load_prompt_template("shared_instructions.txt")
        self._persona = self.load_prompt_template("heuristic_bidder_prompt.txt")

    def build_prompt(self, request: BidRequest, fallback_bid: float) -> str:
        if request.history:
            history_text = json.dumps(request.history, indent=2)
        else:
            history_text = "No history available."

        return f"""
        {self._shared}

        {self._persona}

        bidder_id: "{self.bidder_id}"
        private_value: {request.private_value}
        - current_shading_factor: {self.shading_factor}
        - fallback_bid (using current factor): {fallback_bid}

        Analyze carefully the history of past rounds, with winner_id and payoffs included:
        {history_text}

        Based on your performance in past rounds (wins, losses, payoffs), decide:
        1. Should you adjust your shading factor?
        2. What bid should you submit?
        3. Explain your reasoning for both decisions.

        Return JSON with:
        - "bid": your bid amount
        - "new_shading_factor": your adjusted factor (or keep current if no change)
        - "reasoning": explanation of your decision
        """.strip()

    def get_bid(self, request: BidRequest) -> BidResponse:
        # Always define a deterministic fallback
        fallback_bid = self.shading_factor * request.private_value

        if not self.use_llm:
            # No LLM: just use the heuristic directly
            auction_id = request.auction_id
            round_index = request.round_index

            return BidResponse(
                auction_id=auction_id,
                round_index=round_index,
                bidder_id=self.bidder_id,
                bid=fallback_bid,
                reasoning=f"Heuristic bidder with shading factor {self.shading_factor}.",
                raw_llm_text=None,
            )

        # If LLM is enabled, try to call it and parse JSON
        prompt = self.build_prompt(request, fallback_bid=fallback_bid)

        try:
            raw_text = self.call_llm(prompt)
        except NotImplementedError:
            # LLM not wired yet – fallback
            auction_id = request.auction_id
            round_index = request.round_index

            return BidResponse(
                auction_id=auction_id,
                round_index=round_index,
                bidder_id=self.bidder_id,
                bid=fallback_bid,
                reasoning="LLM not configured, used fallback shading rule.",
                raw_llm_text=None,
            )
        #print(raw_text)
        response = self.parse_bid_response(request, raw_text)

        # Extract and apply new shading factor if provided
        try:
            data = self._extract_json(raw_text)
            if "new_shading_factor" in data:
                new_factor = float(data["new_shading_factor"])
                self.shading_factor = min(max(new_factor, 0.7), 0.95)  # clamp to bounds
        except Exception:
            print("Failed to extract new shading factor from LLM response.")  # If extraction fails, keep current factor

        return response

        '''return self.parse_bid_response(
            self=self,
            raw_text=raw_text,
            fallback_bid=fallback_bid,
            bidder_id=self.bidder_id,
            reasoning_summary=f"Heuristic bidder with shading factor {self.shading_factor}.",
        )'''