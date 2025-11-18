def run_second_price_auction(bids: dict[str, float]) -> tuple[str, float]:
    # returns (winner_id, clearing_price)
        if not bids:
            return None, 0.0

        # Sort bidders by bid, highest first
        sorted_items = sorted(bids.items(), key=lambda x: x[1], reverse=True)

        # Highest bid
        winner_id, highest_bid = sorted_items[0]

        # Second price
        if len(sorted_items) > 1:
            second_price = sorted_items[1][1]
        else:
            # Only one bidder → pays 0 in a standard Vickrey auction
            second_price = 0.0

        return winner_id, second_price
