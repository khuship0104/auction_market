# agents/auctioneer_agent.py

from __future__ import annotations

from typing import Sequence

from core.models import AuctionConfig, BidRequest, BidResponse, AuctionOutcome
from core.value_sampler import sample_value_uniform_0_1
from tools.payoff_calculator import compute_payoffs
from .base_agent import BaseAgent


class AuctioneerAgent(BaseAgent):
    """
    Coordinates one auction round:
    - Samples private values
    - Asks each bidder for a BidResponse
    - Calls payoff calculator to get AuctionOutcome
    """

    def __init__(self, config: AuctionConfig, auction_id: str = "auction_001"):
        super().__init__(name="Auctioneer")
        self.config = config
        self.auction_id = auction_id
        self._round_counter = 0
        self.bid_history = []  # To store bid records

    def get_bid(self, request: BidRequest) -> BidResponse:
        """
        Auctioneer does not bid; this method is not used.
        """
        raise NotImplementedError("Auctioneer does not implement get_bid.")

    def run_round(self, bidders: Sequence[BaseAgent]) -> AuctionOutcome:
        """
        Run a single sealed-bid second-price auction round.
        """
        bids: dict[str, float] = {}
        values: dict[str, float] = {}

        current_round = self._round_counter

        # 1) Sample private values and get bids
        for bidder in bidders:
            bidder_id = getattr(bidder, "bidder_id", bidder.name)
            bid_responses: dict[str, BidResponse] = {}
            v = sample_value_uniform_0_1()
            values[bidder_id] = v

            req = BidRequest(
                auction_id=self.auction_id,
                round_index=current_round,
                auction_config=self.config,
                bidder_id=bidder_id,
                private_value=v,
                history=None,  # or pass a real history object later
            )

            if hasattr(bidder, "get_bid"):
                bid_response: BidResponse = bidder.get_bid(req)
                bid_responses[bidder_id] = bid_response
                bids[bidder_id] = bid_response.bid

                # Append each bidder's bid to bid_history for plotting purposes
                bid_type = "Heuristic" if bidder.name.startswith("HeuristicBidder") else "Strategic"
                self.bid_history.append({
                    "round": current_round,
                    "agent_id": bidder_id,
                    "agent_type": bid_type,
                    "bid": bid_response.bid
                })
            else:
                raise ValueError(f"Bidder {bidder_id} has no get_bid method.")

        # 2) Compute outcome with payoff tool
        outcome: AuctionOutcome = compute_payoffs(
            bids=bids,
            values=values,
            auction_id=self.auction_id,
            round_index=current_round,

        )

        self._round_counter += 1


        return outcome, bid_responses
