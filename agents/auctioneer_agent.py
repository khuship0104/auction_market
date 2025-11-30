# agents/auctioneer_agent.py

from __future__ import annotations

from typing import Sequence

from core.models import AuctionConfig, BidRequest, BidResponse, AuctionOutcome
from core.value_sampler import sample_value_uniform_0_1
from tools.payoff_calculator import compute_payoffs
from .base_agent import BaseAgent
import json


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
        self.summary_file_path = "auction_summaries.txt" # for writing the summaries to file


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
                history=self.build_history(),  # or pass a real history object later
            )

            if hasattr(bidder, "get_bid"):
                bid_response: BidResponse = bidder.get_bid(req)
                bid_responses[bidder_id] = bid_response
                bids[bidder_id] = bid_response.bid

                # Optional: print bid and reasoning for debugging for strategic agents
                debug_response = False
                if bidder.name.startswith("HeuristicBidder") and debug_response:
                    print(f"[Heuristic Agent {bidder_id}] Bid: {bid_response.bid}")
                    #print(f"  Reasoning: {bid_response.reasoning}")
                    # (Optional) print full raw LLM output
                    print(f"  Raw LLM Output:\n{bid_response.raw_text}\n")

                # Append each bidder's bid to bid_history for plotting purposes
                bid_type = "Heuristic" if bidder.name.startswith("HeuristicBidder") else "Strategic"
                self.bid_history.append({
                    "round": current_round,
                    "agent_id": bidder_id,
                    "agent_type": bid_type,
                    "bid": bid_response.bid,
                    "secret_value": v
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

        """
        3) Generate and print LLM-written summary of the round
        Can either opt to print to console or write to a file.
        Default is to write to a file.
        """
        llm_summary = self.generate_round_summary(outcome) # call func for LLM summary
        #print(f"\n Auctioneer Summary (Round {current_round}):\n{llm_summary}\n") # uncomment to print to console
        write_mode = "w" if current_round == 0 else "a" # clear write file on first round
        with open(self.summary_file_path, write_mode, encoding="utf-8") as f:
            f.write(f"Auctioneer Summary (Round {current_round}):\n")
            f.write(llm_summary)
            f.write("\n\n")

        self._round_counter += 1

        return outcome, bid_responses

    def generate_round_summary(self, outcome: AuctionOutcome):
        """
        Generates an LLM-written human-readable summary for a single round, not entire auction.
        """
        prompt = f"""
    You are the auctioneer overseeing a second-price sealed-bid auction.
    Provide a concise, human-readable commentary for ROUND {outcome.round_index}.

    Here is the full AuctionOutcome object:
    {outcome}

    Write a concise, human-readable 2–4 sentence analysis including:
    - The winner and why they won
    - How competitive the bidding was
    - Whether bids appear truthful or shaded
    - How the price compares to valuations
    - Any notable strategic insights

    Avoid JSON. Produce natural language only.
    """

        try:
            summary = self.call_llm(prompt)
        except Exception as e:
            summary = f"[LLM summary unavailable due to error: {e}]"

        return summary

    def build_history(self) -> dict:
        """
        Return a structured history of the last K rounds suitable for StrategicAgent LLM.
        Only includes agent_id, agent_type, bid, and secret_value.
        """
        rounds_history = []

        rounds_to_include = sorted({entry["round"] for entry in self.bid_history})

        for r in rounds_to_include:
            round_entries = [e for e in self.bid_history if e["round"] == r]
            bids_dict = {e["agent_id"]: e["bid"] for e in round_entries}
            secret_dict = {e["agent_id"]: e["secret_value"] for e in round_entries}

            rounds_history.append({
                "round_index": r,
                "bids": bids_dict,
                "secret_values": secret_dict  # optional
            })

        # Optional summary stats
        avg_bid = sum(e["bid"] for e in self.bid_history) / len(self.bid_history) if self.bid_history else 0

        return {
            "rounds": rounds_history,
            "summary_stats": {
                "num_rounds": len(rounds_history),
                "avg_bid": avg_bid
            }
        }
