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
        #self.summary_file_path = "auction_summaries.txt" # for writing the summaries to file, moved to full scenario output files

    def get_bid(self, request: BidRequest) -> BidResponse:
        """
        Auctioneer does not bid; this method is not used.
        """
        raise NotImplementedError("Auctioneer does not implement get_bid.")

    def run_round(self, bidders: Sequence[BaseAgent], log_file: str = None) -> tuple[AuctionOutcome, dict[str, BidResponse]]:
        """
        Run a single sealed-bid second-price auction round.
        
        Args:
            bidders: List of bidder agents
            log_file: Optional file path to log round details
        """
        bids: dict[str, float] = {}
        values: dict[str, float] = {}

        current_round = self._round_counter
        bid_responses: dict[str, BidResponse] = {}
        
        # 1) Sample private values and get bids
        for bidder in bidders:
            bidder_id = getattr(bidder, "bidder_id", bidder.name)
            v = sample_value_uniform_0_1()
            values[bidder_id] = v

            req = BidRequest(
                auction_id=self.auction_id,
                round_index=current_round,
                auction_config=self.config,
                bidder_id=bidder_id,
                private_value=v,
                history=self.build_history(exclude_round=current_round),
            )

            if hasattr(bidder, "get_bid"):
                bid_response: BidResponse = bidder.get_bid(req)
                bid_responses[bidder_id] = bid_response
                bids[bidder_id] = bid_response.bid

                # Append each bidder's bid to bid_history for plotting purposes
                bid_type = "Heuristic" if bidder.name.startswith("HeuristicBidder") else "Strategic"
                bid_record = {
                    "round": current_round,
                    "agent_id": bidder_id,
                    "agent_type": bid_type,
                    "bid": bid_response.bid,
                    "secret_value": v,
                }
                
                # Capture shading factor for heuristic agents
                if bid_type == "Heuristic" and hasattr(bidder, "shading_factor"):
                    bid_record["shading_factor"] = bidder.shading_factor
                
                self.bid_history.append(bid_record)
            else:
                raise ValueError(f"Bidder {bidder_id} has no get_bid method.")

        # 2) Compute outcome with payoff tool
        outcome: AuctionOutcome = compute_payoffs(
            bids=bids,
            values=values,
            auction_id=self.auction_id,
            round_index=current_round,
        )

        # Add winner_id to each bid_history entry for this round
        for entry in self.bid_history:
            if entry["round"] == current_round:
                entry["winner_id"] = outcome.winner_id


        llm_summary = self.generate_round_summary(outcome) # call func for LLM summary

        self._round_counter += 1

        # last step is logging round details in output file
        self.write_round_log(log_file, current_round, bidders, bid_responses, outcome, llm_summary)

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

    def build_history(self, exclude_round: int = None) -> dict:
        """
        Return a structured history, optionally excluding a specific round.
        """
        rounds_history = []
        rounds_to_include = sorted({entry["round"] for entry in self.bid_history})
        
        for r in rounds_to_include:
            if exclude_round is not None and r == exclude_round:
                continue
            round_entries = [e for e in self.bid_history if e["round"] == r]
            bids_dict = {e["agent_id"]: e["bid"] for e in round_entries}
            secret_dict = {e["agent_id"]: e["secret_value"] for e in round_entries}
            winner_id = round_entries[0].get("winner_id") if round_entries else None

            rounds_history.append({
                "round_index": r,
                "bids": bids_dict,
                "secret_values": secret_dict,
                "winner_id": winner_id,
            })

        return {
            "rounds": rounds_history,
            "summary_stats": {
                "num_rounds": len(rounds_history)
            }
        }

    def write_round_log(self, log_file: str, round_num: int, bidders: Sequence[BaseAgent], bid_responses: dict[str, BidResponse], outcome: AuctionOutcome, llm_summary: str) -> None:
        """Write one round's decisions and outcome to log file."""
        mode = "w" if round_num == 0 else "a" # overwrite log file so each sim is fresh
        
        with open(log_file, mode, encoding="utf-8") as f:
            if round_num == 0:
                f.write("AUCTION SIMULATION LOG\n" + "="*60 + "\n\n")
            
            f.write(f"--- ROUND {round_num} ---\n")
            
            # Log each bidder's bid and reasoning
            for bidder in bidders:
                bidder_id = getattr(bidder, "bidder_id", bidder.name)
                if bidder_id in bid_responses:
                    response = bid_responses[bidder_id]
                    f.write(f"{bidder_id}: bid={response.bid:.4f}\n")
                    if response.reasoning:
                        f.write(f"  reasoning: {response.reasoning}\n")
            
            f.write(f"Winner: {outcome.winner_id}, Clearing Price: {outcome.clearing_price:.4f}\n")
            f.write(f"Summary: {llm_summary}\n\n") # auctioneer agents summary