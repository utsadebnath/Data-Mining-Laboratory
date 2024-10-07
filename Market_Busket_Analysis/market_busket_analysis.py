# market busket analysis using apriori_algorithm
import pandas as pd
from collections import defaultdict
from itertools import combinations

# Load the dataset
df = pd.read_csv('HyderabadResturants.csv')

# Preprocess the data to extract cuisines as transactions
df['cuisine'] = df['cuisine'].apply(lambda x: x.split(', '))


# Step 1: Generate candidate k-itemsets and count their occurrences
def get_itemsets(transactions, itemset_size):
    itemset_counts = defaultdict(int)
    for transaction in transactions:
        for itemset in combinations(set(transaction), itemset_size):
            itemset_counts[frozenset(itemset)] += 1
    return itemset_counts


# Step 2: Filter itemsets by minimum support
def filter_itemsets_by_support(itemset_counts, min_support_count):
    return {itemset: count for itemset, count in itemset_counts.items() if count >= min_support_count}


# Step 3: Generate candidate k-itemsets from frequent (k-1)-itemsets
def generate_candidate_itemsets(frequent_itemsets, k):
    candidates = set()
    itemsets = list(frequent_itemsets.keys())

    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            union_itemset = itemsets[i].union(itemsets[j])
            if len(union_itemset) == k:
                candidates.add(union_itemset)

    return candidates


# Step 4: Count occurrences of candidate itemsets
def count_candidate_itemsets(transactions, candidates):
    candidate_counts = defaultdict(int)
    for transaction in transactions:
        transaction_set = set(transaction)
        for candidate in candidates:
            if candidate.issubset(transaction_set):
                candidate_counts[candidate] += 1
    return candidate_counts


# Calculate optimal min_support
def calculate_optimal_min_support(transactions, support_range):
    support_results = {}

    for min_support in support_range:
        min_support_count = min_support * len(transactions)

        all_frequent_itemsets = []
        k = 1

        while True:
            if k == 1:
                itemset_counts_k = get_itemsets(transactions, k)
            else:
                candidate_k_itemsets = generate_candidate_itemsets(frequent_itemsets, k)
                if not candidate_k_itemsets:
                    break
                itemset_counts_k = count_candidate_itemsets(transactions, candidate_k_itemsets)

            frequent_itemsets = filter_itemsets_by_support(itemset_counts_k, min_support_count)

            if not frequent_itemsets:
                break

            all_frequent_itemsets.append(frequent_itemsets)
            k += 1

        total_itemsets = sum(len(itemset) for itemset in all_frequent_itemsets)
        support_results[min_support] = total_itemsets

    optimal_support = max(support_results, key=support_results.get)
    return optimal_support, support_results


# Get transactions from the cuisine column
transactions = df['cuisine'].tolist()

# Define a range of min_support values to test
support_range = [i / 100.0 for i in range(3, 11)]  # 0.03 to 0.1

# Calculate optimal min_support
optimal_support, support_results = calculate_optimal_min_support(transactions, support_range)

# Output the results
print("Support results:")
for support, count in support_results.items():
    print(f"  Min Support: {support:.2f}, Frequent Itemsets Found: {count}")
print(f"Optimal min_support: {optimal_support}")
# (Optional) Output frequent itemsets for the optimal support
min_support_count = optimal_support * len(transactions)
all_frequent_itemsets = []
k = 1

while True:
    if k == 1:
        itemset_counts_k = get_itemsets(transactions, k)
    else:
        candidate_k_itemsets = generate_candidate_itemsets(frequent_itemsets, k)
        if not candidate_k_itemsets:
            break
        itemset_counts_k = count_candidate_itemsets(transactions, candidate_k_itemsets)

    frequent_itemsets = filter_itemsets_by_support(itemset_counts_k, min_support_count)

    if not frequent_itemsets:
        break

    all_frequent_itemsets.append(frequent_itemsets)
    k += 1

# Output all frequent itemsets for the optimal support
for i, itemsets in enumerate(all_frequent_itemsets, start=1):
    print(f"Frequent {i}-itemsets for optimal support {optimal_support}:")
    for itemset, count in itemsets.items():
        print(f"  {set(itemset)}: {count}")