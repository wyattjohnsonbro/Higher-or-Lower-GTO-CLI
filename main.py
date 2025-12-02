import statistics

# Standard card ranks from 2 to Ace. Useful for ordered comparisons.
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Mapping rank to a numeric value. Allows arithmetic comparisons and summary stats.
VALUES = {r: i for i, r in enumerate(RANKS, start=2)}

# Each rank occurs 4 times per deck, two decks combined → 8 copies per rank.
# Float type is intentional for proportional discard calculations.
INITIAL_COUNT = 8.0

# Dictionary holding remaining counts per rank. Will mutate during play.
counts = {r: INITIAL_COUNT for r in RANKS}

# History of played cards for computing observed statistics.
history = []


def total_remaining() -> float:
    """Return total number of remaining cards in the deck (float for probabilistic operations)."""
    return sum(counts.values())


def suggest(card: str) -> None:
    """Compute and display probability of next card being higher/lower than the given card.

    Notes for future ports:
    - Probabilities are computed dynamically using current counts; could be cached for efficiency.
    - Could extend to handle suits if moving to a more granular simulation.
    - Currently assumes replacement after play is not automatic; may need adjustment for different game rules.
    """
    if card not in VALUES:
        print("Invalid card rank. Use one of:", RANKS)
        return

    value = VALUES[card]
    # Compute counts of strictly higher/lower cards
    higher = sum(counts[r] for r in RANKS if VALUES[r] > value)
    lower = sum(counts[r] for r in RANKS if VALUES[r] < value)
    total = max(total_remaining(), 0.0)

    if total <= 0:
        print("No cards left!")
        return

    p_higher = higher / total
    p_lower = lower / total

    print(f"Probability higher: {p_higher:.4f}")
    print(f"Probability lower : {p_lower:.4f}")
    print()  # single blank line

    # Suggest move based purely on probability; ties suggest random choice.
    if p_higher > p_lower:
        print("Go HIGHER")
    elif p_lower > p_higher:
        print("Go LOWER")
    else:
        print("Flip a coin...")


def play_card(card: str) -> None:
    """Mark a card as played, update counts, record in history, and suggest next move.

    Future considerations:
    - Could add undo functionality by popping from history and restoring counts.
    - Could track counts per player if porting to multiplayer simulations.
    """
    if counts.get(card, 0.0) <= 0:
        print(f"No {card}s left in deck!")
        return

    counts[card] -= 1.0
    history.append(card)
    suggest(card)


def discard_unseen(n: int) -> None:
    """Remove n unseen cards probabilistically, maintaining proportional distribution.

    Key points:
    - Fractional reduction keeps the ratios of remaining cards intact.
    - Using float counts allows partial discards; could be rounded for discrete card simulations.
    - Prevents over-discarding by clamping fraction to 1.0.
    """
    remaining = total_remaining()
    if remaining <= 0:
        print("No cards left!")
        return

    if n <= 0:
        print("Must discard a positive number.")
        return

    frac = min(n / remaining, 1.0)
    for r in RANKS:
        counts[r] *= (1.0 - frac)

    print(f"Discarded {n} unseen cards probabilistically.")


def reset() -> None:
    """Reset deck to initial 2-deck shoe and clear history.

    Notes for future enhancements:
    - Could parameterize deck size or number of decks.
    - Could preserve history for analysis rather than clearing it.
    """
    global counts, history
    counts = {r: INITIAL_COUNT for r in RANKS}
    history = []
    print("Deck reset.")


def distribution_table() -> None:
    """Print table of counts, exact probabilities, and cumulative >= probabilities.

    Useful for:
    - Validating that discards preserve proportional probabilities.
    - Providing insight for strategy decisions.
    """
    total = total_remaining()
    if total <= 0:
        print("No cards left!")
        return

    print(f"{'Rank':>4}  {'Count':>8}  {'P(exact)':>10}  {'P(>= rank)':>12}")
    counts_by_value = sorted(RANKS, key=lambda r: VALUES[r], reverse=True)
    cum = 0.0
    ge_prob = {}
    for r in counts_by_value:
        cum += counts[r]
        ge_prob[r] = cum / total

    for r in RANKS:
        cnt = counts[r]
        p_exact = cnt / total
        p_ge = ge_prob[r]
        print(f"{r:>4}  {cnt:8.2f}  {p_exact:10.6f}  {p_ge:12.6f}")

    print(f"\nTotal remaining: {total:.2f}")


def numeric_from_rank_list(rank_list):
    """Convert rank strings to numeric values using VALUES mapping.

    Useful for statistics, plotting, or ML features if extending this simulator.
    """
    return [VALUES[r] for r in rank_list]


def initial_pmf():
    """Return probability mass function for initial deck.

    Can be reused for baseline simulations or predictive models.
    """
    total_init = INITIAL_COUNT * len(RANKS)
    return {r: INITIAL_COUNT / total_init for r in RANKS}


def predicted_single_draw_summary():
    """Return predicted statistics for a single draw from the initial deck.

    Notes:
    - Computes mean, median, variance, and standard deviation.
    - Median rank calculation uses cumulative probability; could be adjusted for even-numbered decks.
    """
    pmf = initial_pmf()
    mean = sum(VALUES[r] * p for r, p in pmf.items())

    cum = 0.0
    median_rank = None
    for r in RANKS:
        cum += pmf[r]
        if cum >= 0.5:
            median_rank = r
            break

    mean_sq = sum((VALUES[r] ** 2) * p for r, p in pmf.items())
    variance_pop = mean_sq - mean ** 2
    std_pop = variance_pop ** 0.5

    return {
        "pmf": pmf,
        "mean": mean,
        "median_rank": median_rank,
        "variance_population": variance_pop,
        "std_population": std_pop,
    }


def predicted_summary():
    """Return predicted summary statistics (mean, median, variance, std) for single draws.

    Excludes min/max/range for simplicity; could be extended for richer analysis.
    """
    single = predicted_single_draw_summary()
    return {
        "mean": single["mean"],
        "median_rank": single["median_rank"],
        "variance_population": single["variance_population"],
        "std_population": single["std_population"],
    }


def observed_summary():
    """Compute statistics from observed history of played cards.

    Notes:
    - Returns sample and population variance/std for flexibility.
    - Range computation useful for monitoring spread, not used in predicted stats.
    - If fewer than 2 draws, sample variance/std set to 0.
    - Could be extended to sliding window or per-player stats in multiplayer ports.
    """
    n = len(history)
    if n == 0:
        return None

    nums = numeric_from_rank_list(history)
    mean = statistics.mean(nums)
    median = statistics.median(nums)
    rng = max(nums) - min(nums)

    if n >= 2:
        stdev_sample = statistics.stdev(nums)
        var_sample = statistics.variance(nums)
    else:
        stdev_sample = 0.0
        var_sample = 0.0

    stdev_pop = statistics.pstdev(nums)
    var_pop = statistics.pvariance(nums)

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "range": rng,
        "stdev_sample": stdev_sample,
        "variance_sample": var_sample,
        "stdev_population": stdev_pop,
        "variance_population": var_pop,
        "nums": nums,
    }


def print_stats_compare():
    """Print predicted vs observed summary statistics side by side.

    Notes:
    - Observed stats dynamically reflect history; predicted stats remain fixed for baseline comparison.
    """
    obs = observed_summary()
    pred = predicted_summary()

    print("=== Predicted ===")
    print(f"Predicted mean: {pred['mean']:.4f}")
    print(f"Predicted median rank: {pred['median_rank']}")
    print(f"Predicted population std: {pred['std_population']:.4f}")
    print(f"Predicted population variance: {pred['variance_population']:.6f}")
    print()

    print("=== Observed ===")
    if obs is None:
        print("No observed plays yet.")
        return

    inv_values = {v: r for r, v in VALUES.items()}
    print(f"Number of draws: {obs['n']}")
    print(f"Observed mean: {obs['mean']:.4f}")
    print(f"Observed median: {obs['median']:.4f}")
    print(f"Observed range: {obs['range']:.4f}")
    if obs['n'] >= 2:
        print(f"Observed sample std: {obs['stdev_sample']:.4f}")
        print(f"Observed sample variance: {obs['variance_sample']:.6f}")
    else:
        print("Observed sample std/variance: need at least 2 draws")
        print(f"Observed population std: {obs['stdev_population']:.4f}")
        print(f"Observed population variance: {obs['variance_population']:.6f}")


# Main loop
print("Higher or Lower!!!!!")
print("Commands:")
print("  - Enter a card rank 2–A to play that card.")
print("  - Enter 'discard n' to remove n unseen cards.")
print("  - Enter 'show' to see remaining rank counts.")
print("  - Enter 'table' to see probability table.")
print("  - Enter 'stats' to compare predicted vs observed summary statistics.")
print("  - Enter 'reset' to restart.")
print("  - Enter 'quit' to exit.")
print()

while True:
    user_input = input("Enter card or command: ").strip().upper()

    if user_input == "QUIT":
        break
        
    if user_input == "RESET":
        reset()
        continue
        
    if user_input == "SHOW":
        for r in RANKS:
            print(f"{r}: {counts[r]:.2f}", end="  ")
        print(f"\nTotal: {total_remaining():.2f}")
        continue
        
    if user_input == "TABLE":
        distribution_table()
        continue
        
    if user_input == "STATS":
        print_stats_compare()
        continue
        
    if user_input.startswith("DISCARD"):
        parts = user_input.split()
        if len(parts) == 2 and parts[1].isdigit():
            discard_unseen(int(parts[1]))
        else:
            print("Usage: discard <number>")
        continue
        
    if user_input in RANKS:
        play_card(user_input)
        continue

    print("Invalid input. Try a rank 2..A or one of the commands.")
