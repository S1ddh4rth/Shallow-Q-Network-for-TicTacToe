import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    """
    Parses the log file to extract training details.

    Parameters:
    log_file_path (str): Path to the log file.

    Returns:
    tuple: A tuple containing lists for episodes, outcomes, average performance, and smartMovePlayer1 increments.
    """
    episodes = []
    avg_performance = []
    outcomes = {"win": 0, "loss": 0, "draw": 0}
    outcome_per_episode = []
    smartMove_increments = []

    with open(log_file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if "Increment at Episode" in line:
            # Parse increment points
            try:
                episode = int(line.split(" ")[3].strip(":"))  # Extract episode number correctly
                smartMove_increments.append(episode)
            except ValueError:
                print(f"Skipping invalid increment line: {line}")

        elif line.strip() and not line.startswith("Episode"):
            # Parse regular episode logs
            parts = line.strip().split(",")
            try:
                episode = int(parts[0])
                outcome = parts[1].strip()
                avg_perf = float(parts[2])

                episodes.append(episode)
                avg_performance.append(avg_perf)

                # Tally outcomes
                if outcome == "2":
                    outcomes["win"] += 1
                    outcome_per_episode.append("win")
                elif outcome == "1":
                    outcomes["loss"] += 1
                    outcome_per_episode.append("loss")
                else:
                    outcomes["draw"] += 1
                    outcome_per_episode.append("draw")
            except (IndexError, ValueError):
                print(f"Skipping invalid log line: {line}")

    # print(smartMove_increments)
    return episodes, avg_performance, outcomes, outcome_per_episode, smartMove_increments



def plot_training_metrics(log_file_path):
    """
    Plots training metrics based on the log file.

    Parameters:
    log_file_path (str): Path to the log file.
    """
    (
        episodes,
        avg_performance,
        outcomes,
        outcome_per_episode,
        smartMove_increments,
    ) = parse_log_file(log_file_path)

    import matplotlib.pyplot as plt

    # Plot 1: Wins, Losses, and Draws
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(["Wins", "Losses", "Draws"], outcomes.values(), color=["green", "red", "blue"])
    ax1.set_title("Game outcomes during training", fontsize=16)
    ax1.set_ylabel("Count", fontsize=20)
    ax1.set_xlabel("Outcome", fontsize=20)
    ax1.tick_params(axis='both', labelsize=12)  # Increase tick label size
    plt.tight_layout()
    plt.savefig("img1.png")
    plt.savefig("img1.eps")

    # Plot 2: Avg Performance with SmartMovePlayer1 increments
    fig, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(episodes, avg_performance, label="Avg Performance", color="royalblue", linewidth=2)
    ax2.set_ylabel("Avg Performance over previous 50 episodes", fontsize=16)
    ax2.set_xlabel("Episode number", fontsize=20)
    # ax2.set_title("Average performance over previous 50 episodes", fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)  # Increase tick label size
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Add markers for SmartMovePlayer1 increments
    ind = 0
    for increment in smartMove_increments:
        if ind == 0:
            ax2.axvline(x=increment, color="orange", linestyle="--", label="+0.1 to smartness")
        else:
            ax2.axvline(x=increment, color="orange", linestyle="--")
        ind += 1

    # Remove duplicate labels for increment lines
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys(), fontsize=12)

    plt.tight_layout()
    plt.savefig("img2.png")
    plt.savefig("img2.eps")



if __name__ == "__main__":
    log_file = "episode_log2.txt"
    plot_training_metrics(log_file)
