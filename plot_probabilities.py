import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from collections import defaultdict

df = pd.read_csv("new_4-block_monomers.csv")
prob_of_improvements = df["Probability of Improvement"].values

def at_least(n_successes, n_possible, probabilities):
    """ Find probability of at least n_successes successful monomers out of n_possible attempted """
    # We will find 1 - probability of at most (n_successes - 1) successes
    prob = 0
    possible_indices = np.asarray(list(range(n_possible)))  # possible monomers to choose from
    for i in range(n_successes):
        # Get all combinations of (n_possible - i) failures
        failing_indices = np.asarray(list(combinations(list(range(n_possible)), n_possible - i)))

        # Find the indices in possible_indices that are not in failing_indices
        successful_indices = np.asarray([np.setdiff1d(possible_indices, f) for f in failing_indices])

        failures = 1 - probabilities[[failing_indices]]  # Get probabiliies of monomers failing
        successes = probabilities[[successful_indices]]  # Get probabiliies of monomers succeeding
        combined_probs = np.hstack((failures, successes))  # Combine probabilities into one row

        # Multply across rows to get probability of exact sequence of successes and failures
        # Add all possible sequences of the same number of successes and failures
        prob += np.sum(np.prod(combined_probs, axis=1))

    return 1 - prob  # return 1 - probability of at most (n_successes - 1) successes

# Create lists of probabilities to plot
probs_dict = defaultdict(list)
n_possible_dict = defaultdict(list)
for i in range(1, 4):
    # Find probability of 1, 2, and 3 successes
    print()
    for j in range(i, 50):
        # Find probabilities when considering [i, 49] possible monomers
        print(j)
        probs_dict[i].append(at_least(i, j, prob_of_improvements)) # calculate probability
        n_possible_dict[i].append(j)

# Plot probability curves
plt.figure()
legend_entries = []
for n_successes, probabilities in probs_dict.items():
    plt.plot(n_possible_dict[n_successes], probabilities)
    legend_str = str(n_successes) + " successes"
    if n_successes == 1:
        legend_str = legend_str[:-2]
    legend_entries.append(legend_str)
plt.legend(legend_entries)
plt.xlabel("Number of Monomers Attempted", fontsize=14)
plt.ylabel("Probability of Finding Successful Monomers", fontsize=14)
plt.show()
