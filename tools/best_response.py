import random
from tools.payoff_calculator import compute_payoffs  # assumes this returns AuctionOutcome


def approximate_best_response(private_value: float, num_grid_points: int = 101) -> dict:
    """
    Approximate a best-response bid in a second-price auction against
    (by default) truthful, uniformly distributed opponents in [0, 1].

    Uses the payoff calculator internally to evaluate expected utility.
    """

    rng = random.Random(0)  # fixed seed for reproducibility

    player_id = "me"
    NUM_OPPONENTS = 2         # assume a 2-bidder auction (you + 2 opponents)
    SAMPLES_PER_BID = 500     # Monte Carlo samples per candidate bid

    # Grid of candidate bids in [0, 1]
    if num_grid_points < 2:
        raise ValueError("num_grid_points must be at least 2")
    grid = [i / (num_grid_points - 1) for i in range(num_grid_points)]

    best_bid = 0.0
    best_utility = float("-inf")

    for b in grid:
        total_utility = 0.0

        for _ in range(SAMPLES_PER_BID):
            # Sample opponent values ~ U[0, 1] and assume truthful bidding for them
            values = {player_id: private_value}
            bids = {player_id: b}

            for j in range(NUM_OPPONENTS):
                opp_id = f"opp_{j}"
                opp_value = rng.random()
                values[opp_id] = opp_value
                bids[opp_id] = opp_value  # truthful opponent

            outcome = compute_payoffs(bids=bids, values=values)
            total_utility += outcome.payoffs[player_id]

        avg_utility = total_utility / SAMPLES_PER_BID

        if avg_utility > best_utility:
            best_utility = avg_utility
            best_bid = b

    return {"best_bid": float(best_bid), "expected_utility": float(best_utility)}