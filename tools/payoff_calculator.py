# tools/payoff_calculator.py

from typing import Dict
from core.game_logic import run_second_price_auction
from core.models import AuctionOutcome

def compute_payoffs(
    bids: Dict[str, float],
    values: Dict[str, float],
    auction_id: str = "simulated_auction",
    round_index: int = 0,
) -> AuctionOutcome:
    """
    Run a second-price auction and compute quasilinear payoffs.

    Assumes:
    - run_second_price_auction(bids) -> (winner_id, price)
    - Payoff for winner i:  u_i = v_i - price
    - Payoff for losers:    u_j = 0
    """
    winner_id, clearing_price = run_second_price_auction(bids)

    payoffs: dict[str, float] = {}
    for bidder_id, v in values.items():
        if bidder_id == winner_id:
            payoffs[bidder_id] = v - clearing_price
        else:
            payoffs[bidder_id] = 0.0

    return AuctionOutcome(
        auction_id=auction_id,
        round_index=round_index,
        winner_id=winner_id,
        clearing_price=clearing_price,
        revenue=clearing_price,  # for a single-item auction, revenue = price
        bids=bids,
        values=values,
        payoffs=payoffs,
    )

