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
        based on {history_text}, use adjust_shading_factor to decide new shading factor if needed.
        fallback_bid (from shading rule): {fallback_bid} (use this ONLY if you choose not to modify)"
        """.strip()

    def get_bid(self, request: BidRequest) -> BidResponse:
        # Always define a deterministic fallback
        self.shading_factor = self.adjust_shading_from_history(request.history)
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
                reasoning=f"Used fallback shading rule.",
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

    def adjust_shading_from_history(self, history: Optional[dict]) -> float:
        """
        Returns an adjusted shading factor based on past auction history.
        Default is self.shading_factor if no change needed.
        """
        if not history or "rounds" not in history or len(history["rounds"]) == 0:
            return self.shading_factor  # no history, keep original

        # Construct a simple prompt for LLM or rule-based adjustment
        prompt = f"""
        You are a heuristic bidder.
        Current shading factor: {self.shading_factor}
        History of past rounds:
        {json.dumps(history, indent=2)}

        Suggest a new shading factor between 0.7 and 0.95.
        Return only a number.
        """

        try:
            new_shade_text = self.call_llm(prompt)
            new_shade = float(new_shade_text.strip())
            # clamp to reasonable bounds
            new_shade = min(max(new_shade, 0.7), 0.95)
            return new_shade
        except Exception:
            return self.shading_factor  # fallback if LLM fails