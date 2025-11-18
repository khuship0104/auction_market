# agents/heuristic_bidder_agent.py

from __future__ import annotations

from typing import Optional

from core.models import BidRequest, BidResponse
from .base_agent import BaseAgent


class HeuristicBidderAgent(BaseAgent):
    """
    LLM-backed heuristic bidder.

    - Uses a simple shading rule as a fallback.
    - Optionally calls an LLM to wrap that rule in "reasoning".
    """

    def __init__(self, bidder_id: str, shading_factor: float = 0.8, use_llm: bool = False):
        super().__init__(name=f"HeuristicBidder-{bidder_id}")
        self.bidder_id = bidder_id
        self.shading_factor = shading_factor
        self.use_llm = use_llm

        # Pre-load templates if you want
        self._shared = self.load_prompt_template("shared_instructions.txt")
        self._persona = self.load_prompt_template("heuristic_bidder_prompt.txt")

    def build_prompt(self, request: BidRequest, fallback_bid: float) -> str:
        history_text = request.history_summary if hasattr(request, "history_summary") else "No history provided."

        return f"""
        {self._shared}

        {self._persona}

        bidder_id: "{self.bidder_id}"
        private_value: {request.private_value}
        suggested_fallback_bid: {fallback_bid}
        history: "{history_text}"
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
                reasoning=f"Used heuristic shading factor {self.shading_factor}.",
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
        
        return self.parse_bid_response(
            request,
            raw_text
        )

        '''return self.parse_bid_response(
            self=self,
            raw_text=raw_text,
            fallback_bid=fallback_bid,
            bidder_id=self.bidder_id,
            reasoning_summary=f"Heuristic bidder with shading factor {self.shading_factor}.",
        )'''
