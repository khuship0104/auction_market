# core/models.py
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel


class AuctionConfig(BaseModel):
    """Configuration for a single auction / simulation run."""

    num_bidders: int
    min_value: float = 0.0
    max_value: float = 1.0
    reserve_price: float = 0.0
    mechanism: str = "second_price"  # e.g. "second_price", "first_price"
    value_distribution: str = "uniform"  # descriptive label only
    random_seed: Optional[int] = None


class BidRequest(BaseModel):
    """
    Message sent from the auctioneer to a bidder agent,
    telling it what to bid.
    """

    auction_id: str
    round_index: int
    auction_config: AuctionConfig
    bidder_id: str
    private_value: float
    context: Optional[str] = None  # free-form description for the LLM agent


class BidResponse(BaseModel):
    """Bid returned by a bidder agent in response to a BidRequest."""

    auction_id: str
    round_index: int
    bidder_id: str
    bid: float
    reasoning: Optional[str] = None  # optional explanation from the LLM agent
    raw_text: Optional[str] = None


class AuctionOutcome(BaseModel):
    """Result of running one auction and computing payoffs."""

    auction_id: str
    round_index: int
    winner_id: Optional[str]  # None if no sale (e.g. reserve not met)
    clearing_price: float
    revenue: float  # usually equal to clearing_price, but kept explicit

    # Per-bidder data
    bids: Dict[str, float]    # bidder_id -> submitted bid
    values: Dict[str, float]  # bidder_id -> private value
    payoffs: Dict[str, float] # bidder_id -> utility/payoff


class SimulationSummary(BaseModel):
    """
    Aggregate statistics over many simulated auctions.
    """

    config: AuctionConfig
    num_rounds: int

    mean_revenue: float
    mean_utility_per_bidder: Dict[str, float]          # bidder_id -> average utility
    distribution_of_winners: Dict[str, float]          # bidder_id -> win frequency

    revenue_series: Optional[List[float]] = None       # one value per round (optional)
