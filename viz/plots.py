import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_all_plots(bid_history,summary):
    """
    Generate all relevant plots from bid history.

    Parameters
    ----------
    bid_history : list[dict]
        Each dict contains:
            - "round": int
            - "agent_id": str
            - "agent_type": str
            - "bid": float

    summary : SimulationSummary
        Each summary contains:


    comment out the plots you don't want to generate while testing.
    """
    plot_bid_histogram(bid_history)
    plot_bid_lines(bid_history)
    plot_wins_per_agent(bid_history, summary)
    plot_utility_per_bidder(summary)

def plot_bid_histogram(bid_records):
    """
    Plot a histogram of bids, color-coded by agent type (Strategic vs Heuristic).

    Parameters
    ----------
    bid_records : list[dict]
        Each dict should have:
            - "round": int
            - "agent_id": str
            - "agent_type": str (Strategic or Heuristic)
            - "bid": float
    """
    df = pd.DataFrame(bid_records)
    bins = np.arange(0, 1.1, 0.1) # bin by every 10th percentile
    plt.figure(figsize=(10, 6))

    ax = sns.histplot(
        data=df,
        x="bid",
        hue="agent_type", # color by agent type
        bins=bins,
        palette={"Strategic": "blue", "Heuristic": "orange"},
        edgecolor="black",
        alpha=0.7,
        multiple="layer"
    )

    # Annotate each bin with mean bid per agent type
    agent_types = df["agent_type"].unique()
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for agent_type in agent_types:
        sub_df = df[df["agent_type"] == agent_type]
        for left, right, center in zip(bins[:-1], bins[1:], bin_centers):
            bin_data = sub_df[(sub_df["bid"] >= left) & (sub_df["bid"] < right)]
            if len(bin_data) > 0:
                mean_bid = bin_data["bid"].mean()
                # y = height of the bar
                y = len(bin_data)
                ax.annotate(
                    f"{mean_bid:.2f}",
                    (center, y),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=90
                )

    plt.title("Distribution of Bids by Agent Type")
    plt.xlabel("Bid Amount")
    plt.ylabel("Number of Bids")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_bid_lines(bid_history):
    """
    Line plot of bids per agent across auction rounds.
    
    Parameters
    ----------
    bid_history : list[dict]
        Each dict contains:
            - "round": int
            - "agent_id": str
            - "agent_type": str
            - "bid": float
    """
    df = pd.DataFrame(bid_history)
    if df.empty:
        raise ValueError("No bid data to plot.")

    plt.figure(figsize=(12, 6))

    for agent_id, agent_df in df.groupby("agent_id"):
        plt.plot(
            agent_df["round"],
            agent_df["bid"],
            marker='o',
            label=f"{agent_id} ({agent_df['agent_type'].iloc[0]})"
        )

    # Overlay secret value for winners only
    winner_secret_values = []
    winner_rounds = []

    for r in df["round"].unique():
        round_df = df[df["round"] == r]
        winner_row = round_df.loc[round_df["bid"].idxmax()]
        winner_secret_values.append(winner_row["secret_value"])
        winner_rounds.append(r)

    plt.scatter(
        winner_rounds,
        winner_secret_values,
        marker='x',
        s=100,
        color='red',
        label="Winner Secret Value"
    )

    plt.title("Bid Trajectories per Agent Across Auction Rounds")
    plt.xlabel("Auction Round")
    plt.ylabel("Bid Amount")
    plt.xticks(df["round"].unique())
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_wins_per_agent(bid_history, summary):
    """
    Plot number of wins per agent.

    Parameters
    ----------
    auction_outcomes : list[dict]
        Each dict should have at least:
            - "winner_id": str
            - "round": int
            - Optional: "agent_type": str
    """
    if not hasattr(summary, "distribution_of_winners"):
        raise ValueError("Summary object missing 'distribution_of_winners'.")

    dist = summary.distribution_of_winners
    if not dist:
        raise ValueError("distribution_of_winners is empty.")

    
    agent_type_map = {bid["agent_id"]: bid["agent_type"] for bid in bid_history}

    # Sort agents by fraction of wins
    sorted_agents = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    agent_ids, win_fractions = zip(*sorted_agents)

    # Build x-axis labels with agent type
    labels = [f"{agent_id} - {agent_type_map.get(agent_id, 'Unknown')}" for agent_id in agent_ids]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, win_fractions, color="blue", edgecolor="black")

    plt.title("Fraction of Wins per Agent")
    plt.xlabel("Agent (Bidder Type)")
    plt.ylabel("Fraction of Wins")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_utility_per_bidder(summary):
    """
    Plot mean utility per bidder.

    Parameters
    ----------
    summary : SimulationSummary
        Each summary contains:
            - mean_utility_per_bidder: dict[str, float]
    """
    if not hasattr(summary, "mean_utility_per_bidder"):
        raise ValueError("Summary object missing 'mean_utility_per_bidder'.")

    util = summary.mean_utility_per_bidder
    if not util:
        raise ValueError("mean_utility_per_bidder is empty.")

    # Sort agents by mean utility
    sorted_agents = sorted(util.items(), key=lambda x: x[1], reverse=True)
    agent_ids, mean_utilities = zip(*sorted_agents)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(agent_ids, mean_utilities, color="green", edgecolor="black")

    plt.title("Mean Utility per Bidder")
    plt.xlabel("Agent")
    plt.ylabel("Mean Utility")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()