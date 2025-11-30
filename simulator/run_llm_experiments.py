# simulator/run_llm_experiments.py

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

from core.models import AuctionConfig, SimulationSummary
from agents.heuristic_bidder_agent import HeuristicBidderAgent
from agents.strategic_bidder_agent import StrategicBidderAgent
from agents.auctioneer_agent import AuctioneerAgent
from viz.plots import run_all_plots



def run_llm_experiments(
    num_rounds: int = 100,
    use_llm_for_heuristics: bool = False,
) -> SimulationSummary:
    """
    Run num_rounds auctions with LLM-based bidders and return summary stats.

    B1: heuristic
    B2: strategic (LLM-based)
    B3: heuristic
    """

    # --- 1) Set up config and agents ---

    config = AuctionConfig(
        auction_type="second_price",
        num_bidders=3,
        value_distribution="uniform_0_1",
    )

    bidders = [
        HeuristicBidderAgent("B1", shading_factor=0.8, use_llm=use_llm_for_heuristics),
        StrategicBidderAgent("B2", use_llm=True),
        HeuristicBidderAgent("B3", shading_factor=0.7, use_llm=use_llm_for_heuristics),
    ]

    auctioneer = AuctioneerAgent(config=config)

    # --- 2) Logging containers ---

    revenues: List[float] = []
    total_utility: Dict[str, float] = defaultdict(float)
    wins_count: Dict[str, int] = defaultdict(int)

    # --- 3) Main simulation loop ---

    for _ in range(num_rounds):
        outcome, bid_responses = auctioneer.run_round(bidders)

        # revenue in second-price = clearing_price
        revenues.append(outcome.clearing_price)

        # track winner counts
        winner = outcome.winner_id
        wins_count[winner] += 1

        # accumulate payoffs per bidder
        for bidder_id, u in outcome.payoffs.items():
            total_utility[bidder_id] += u

    # --- 4) Aggregate stats into SimulationSummary ---

    mean_revenue = sum(revenues) / len(revenues) if revenues else 0.0

    mean_utility_per_bidder = {
        bidder_id: total_utility[bidder_id] / num_rounds for bidder_id in total_utility
    }

    distribution_of_winners = {
        bidder_id: wins_count[bidder_id] / num_rounds for bidder_id in wins_count
    }

    summary = SimulationSummary(
        config=config,
        num_rounds=num_rounds,
        mean_revenue=mean_revenue,
        mean_utility_per_bidder=mean_utility_per_bidder,
        distribution_of_winners=distribution_of_winners,
    )
    run_all_plots(auctioneer.bid_history, summary)

    return summary


def main():
    """
    Simple CLI entrypoint: run experiments and print a short summary.
    """
    num_rounds = 20  

    print(f"\nRunning {num_rounds} LLM-based auctions...\n")

    summary = run_llm_experiments(
        num_rounds=num_rounds,
        use_llm_for_heuristics=True,
    )

    # Header
    print("=== LLM Auction Simulation Summary ===\n")

    # Revenue
    print(f"Mean Clearing Price (Revenue): {summary.mean_revenue:.4f}\n")

    # Utilities
    print("Mean Utility per Bidder:")
    for bidder_id, util in summary.mean_utility_per_bidder.items():
        print(f"  • {bidder_id}: {util:.4f}")
    print()

    # Winner distribution
    print("Winner Distribution:")
    for bidder_id, freq in summary.distribution_of_winners.items():
        print(f"  • {bidder_id}: {freq*100:.1f}%")

    print("\n======================================\n")


if __name__ == "__main__":
    main()
