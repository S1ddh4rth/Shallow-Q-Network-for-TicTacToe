import matplotlib.pyplot as plt
import re


def parse_results(file_path):
    """
    Parses the test results from the log file.
    
    Args:
        file_path (str): Path to the test results file.
    
    Returns:
        dict: A dictionary where keys are smartness levels, and values are dictionaries
              containing counts for 'Wins', 'Draws', and 'Losses'.
    """
    results = {}
    current_smartness = None

    with open(file_path, "r") as file:
        for line in file:
            smartness_match = re.match(r"After training at smartness (\d+\.\d+):", line)
            if smartness_match:
                current_smartness = float(smartness_match.group(1))
                results[current_smartness] = {"Wins": 0, "Draws": 0, "Losses": 0}
                continue

            final_test_match = re.match(r"Final test at smartness (\d+\.\d+):", line)
            if final_test_match:
                current_smartness = float(final_test_match.group(1))
                results[current_smartness] = {"Wins": 0, "Draws": 0, "Losses": 0}
                continue

            if "Wins:" in line and "Draws:" in line and "Losses:" in line:
                counts = list(map(int, re.findall(r"\d+", line)))
                if current_smartness is not None:
                    results[current_smartness]["Wins"] = counts[0]
                    results[current_smartness]["Draws"] = counts[1]
                    results[current_smartness]["Losses"] = counts[2]

    return results


def plot_results(results):
    """
    Plots the results using Matplotlib.
    
    Args:
        results (dict): A dictionary with smartness levels and corresponding results.
    """
    smartness_levels = sorted(results.keys())
    wins = [results[smartness]["Wins"] for smartness in smartness_levels]
    draws = [results[smartness]["Draws"] for smartness in smartness_levels]
    losses = [results[smartness]["Losses"] for smartness in smartness_levels]

    plt.figure(figsize=(10, 6))

    plt.plot(smartness_levels, wins, label="Wins", marker="o", color="green")
    plt.plot(smartness_levels, draws, label="Draws", marker="o", color="blue")
    plt.plot(smartness_levels, losses, label="Losses", marker="o", color="red")

    plt.title("TicTacToe Training Results")
    plt.xlabel("Smartness Level")
    plt.ylabel("Number of Outcomes")
    plt.legend()
    plt.grid(True)
    plt.savefig("Results2.png")
    plt.savefig("Results2.eps")


def main():
    log_file = "testing_new_model.txt"
    results = parse_results(log_file)
    plot_results(results)


if __name__ == "__main__":
    main()
