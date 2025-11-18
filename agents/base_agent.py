# agents/base_agent.py

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Optional, Any

import requests

from core.models import BidRequest, BidResponse
from config.openai_client import client, DEFAULT_MODEL  

class BaseAgent(ABC):
    """
    Base class for all agents (Auctioneer + Bidders).

    - Provides shared prompt loading
    - Provides a placeholder call_llm you can later hook to OpenAI / other
    - Provides a helper to parse BidResponse JSON
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_bid(self, request: BidRequest) -> BidResponse:
        """
        For bidder agents: given a BidRequest, return a BidResponse.
        AuctioneerAgent won't implement this (it will have other methods).
        """
        raise NotImplementedError

    # ---------- Prompt loading ----------

    def load_prompt_template(self, filename: str) -> str:
        """
        Load a prompt template from the prompts/ directory.
        """
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        path = prompts_dir / filename
        return path.read_text(encoding="utf-8")

    # ---------- LLM call (stub) ----------

    def call_llm(self, prompt: str, tools: Optional[list[Any]] = None) -> str:
        """
        Call the OpenAI Responses API with a simple prompt.

        Returns the *text* from the model's message, which we expect
        to be a JSON object string for bidders.
        """
        # For now we ignore `tools` — you're passing tool *results* via prompt text
        # If you later want real tool calling, you can extend this.

        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=prompt,
        )

        # The content is a list of "content blocks"; we want the text of the first one
        # See: https://platform.openai.com/docs/guides/text
        message = response.output[0]  # first output
        # Each output has content blocks (e.g., text, tool_call, etc.)
        # We assume the first content block is plain text here.
        first_block = message.content[0]
        text = first_block.text  # this is actually a Typed object
        # Convert that to a plain string
        raw_text = text

        return raw_text

    # ---------- JSON parsing helper ----------

    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        Robustly extract JSON from LLM text.
        Assumes there is at least one {...} block.
        """
        text = text.strip()
        # If it's already plain JSON
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        # Fallback: find first '{' and last '}' and parse that slice
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])

        raise ValueError(f"Could not find JSON in LLM response: {text!r}")
    

    def parse_bid_response(
        self, 
        request: BidRequest, 
        raw_text: str
    ) -> BidResponse:
        """
        Try to parse JSON from the model. If it fails, fall back to a
        simple BidResponse with the given fallback bid.
        """
        data = self._extract_json(raw_text)

        # support either "bid" or "bid_amount"
        bid_val = data.get("bid")
        if bid_val is None:
            bid_val = data.get("bid_amount")

        if bid_val is None:
            raise ValueError(f"LLM response did not contain 'bid' or 'bid_amount': {data}")

        return BidResponse(
            auction_id=request.auction_id,
            round_index=request.round_index,
            bidder_id=request.bidder_id,   # trust our own ID, not the LLM's
            bid=float(bid_val),
            reasoning=data.get("reasoning"),
            raw_text=raw_text,
        )
