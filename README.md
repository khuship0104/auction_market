# AGENTS
    1. Auctioneer Agent
        - coordinates each auction round
        - samples private values for bidders
        - sends BidRequests to each agent
        - collects BidResponses
        - calls payoff calculator
        - returns AuctionOutcome

    2. Strategic Bidder Agent
        - receives its private value from auctioneer
        - cals best-response approximator to compute a recommended bid
        - passes recommendation into LLM prompt
        - LLM returns JSON object with final bid and reasoning

    3. Heuristic Bidder Agent
        - shading rule: bids a simple fraction of a value (bid = 0.8 * private_value)
            - LLM is able to adjust this factor based on auction history, explaining why.
        - represents a non-strategic, rule-of-thumb bidder

    4. Base Agent
        - parent class for agents
        - handles loading prompts, making LLM calls, and parsing JSON output from the LLM


# TOOLS
    1. Payoff Calculator
        - Takes bids and values.
        - Computes winner, clearing price (second-highest bid), and payoffs.
        - Payoff = value − clearing_price for the winner, else 0.

    2. Best-Response Approximator
        - For a given private value, simulates many fake auctions.
        - Computes expected utility for a grid of possible bids.
        - Returns the best bid and its associated expected utility.
        - StrategicBidderAgent uses this output in its LLM reasoning.
    
    3. Simulator Tool
        - Runs many auctions in sequence.
        - Aggregates statistics across rounds: revenue, bidder utilities, and winner frequencies.
        - Used for the final experiments and analysis.

# CORE FILES
    1. models.py
        - Contains Pydantic data models:
            AuctionConfig
            BidRequest
            BidResponse
            AuctionOutcome
            SimulationSummary
        - Ensures strict validation of data passed between agents, tools, and the simulation.

    2. game_logic.py
        - Holds the core mechanics of the auction.
        - Implements second-price (Vickrey) auction rules:
            Identify the highest bid (winner).
            Identify the second-highest bid (clearing price).

    3. value_sampler.py
        - Samples private values for bidders.
        - Currently uses Uniform(0,1)

# SETUP INSTRUCTIONS
    1. Install dependencies

        pip install -r requirements.txt

    2. Create a .env file in the project root

        OPENAI_API_KEY=sk-xxxxx

    3. Make sure config/openai_client.py loads API key and initializes the client

# RUNNING SIMULATIONS/EXPERIMENTS

    Run the following in the project root:

        python3 -m simulator.run_llm_experiments

# EXAMPLE OUTPUT

    Running 20 LLM-based auctions...

    === LLM Auction Simulation Summary ===

    Mean Clearing Price (Revenue): 0.5025

    Mean Utility per Bidder:
    • B1: 0.4842
    • B2: 0.6134
    • B3: 0.5624

    Winner Distribution:
    • B3: 35.0%
    • B2: 55.0%
    • B1: 10.0%

    ======================================

# HOW IT WORKS
    1. On each round:
        - The agent gets its private value.
        - It calls the Best-Response Tool:
            Computes the recommended bid.
            Computes expected utility.
        - These tool outputs are inserted into the LLM prompt.
    2. The LLM takes this numeric information and generates a JSON response containing:

            {
            "bidder_id": "...",
            "bid": <float>,
            "reasoning": "..."
            }

    3. The JSON is validated using Pydantic.
    4. If the LLM output is invalid, a fallback bid (the tool recommendation) is used.
    
    This ensures the LLM is not just guessing, but is grounded in real calculations.

