# tools/simulator_tool.py
from __future__ import annotations

import random
from typing import Dict

from core.models import AuctionConfig, SimulationSummary
from tools.payoff_calculator import compute_payoffs 


def run_simulation(
    auction_config: AuctionConfig,
    strategy_profiles: Dict[str, str],
    num_rounds: int,
) -> SimulationSummary:
    """
    Run many auctions under simple hard-coded strategies.

    strategy_profiles: bidder_id -> strategy name
        strategy in {"truthful", "shaded_0.8"}
    """

    bidder_ids = list(strategy_profiles.keys())

    # Optional sanity check (comment out if annoying)
    assert auction_config.num_bidders == len(
        bidder_ids
    ), "auction_config.num_bidders must match number of strategy_profiles"

    # RNG (use fixed seed if provided for reproducibility)
    rng = random.Random(auction_config.random_seed) if auction_config.random_seed is not None else random

    total_revenue = 0.0
    total_utility_per_bidder: Dict[str, float] = {bidder_id: 0.0 for bidder_id in bidder_ids}
    win_counts: Dict[str, int] = {bidder_id: 0 for bidder_id in bidder_ids}
    revenue_series = []

    for round_idx in range(num_rounds):
        # 1) Sample private values (for now: uniform on [min_value, max_value])
        values: Dict[str, float] = {
            bidder_id: rng.uniform(auction_config.min_value, auction_config.max_value)
            for bidder_id in bidder_ids
        }

        # 2) Generate bids using the chosen strategy
        bids: Dict[str, float] = {}
        for bidder_id in bidder_ids:
            v = values[bidder_id]
            strategy = strategy_profiles[bidder_id]

            if strategy == "truthful":
                bid = v
            elif strategy == "shaded_0.8":
                bid = 0.8 * v
            else:
                # default fallback: truthful
                bid = v

            bids[bidder_id] = bid

        # 3) Run auction + payoffs using the payoff calculator
        outcome = compute_payoffs(
            bids=bids, 
            values=values,
            auction_id=f"sim_round_{round_idx}",
            round_index=round_idx,
        )

        # 4) Aggregate stats
        total_revenue += outcome.revenue
        revenue_series.append(outcome.revenue)

        for bidder_id, payoff in outcome.payoffs.items():
            total_utility_per_bidder[bidder_id] += payoff

        if outcome.winner_id is not None:
            win_counts[outcome.winner_id] += 1

    # 5) Compute means and winner distribution
    if num_rounds > 0:
        mean_revenue = total_revenue / num_rounds
        mean_utility_per_bidder = {
            bidder_id: total_utility_per_bidder[bidder_id] / num_rounds
            for bidder_id in bidder_ids
        }
        distribution_of_winners = {
            bidder_id: win_counts[bidder_id] / num_rounds
            for bidder_id in bidder_ids
        }
    else:
        mean_revenue = 0.0
        mean_utility_per_bidder = {bidder_id: 0.0 for bidder_id in bidder_ids}
        distribution_of_winners = {bidder_id: 0.0 for bidder_id in bidder_ids}

    return SimulationSummary(
        config=auction_config,
        num_rounds=num_rounds,
        mean_revenue=mean_revenue,
        mean_utility_per_bidder=mean_utility_per_bidder,
        distribution_of_winners=distribution_of_winners,
        #revenue_series=revenue_series,
    )
