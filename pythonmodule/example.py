import numpy as np
from _dbscan import DBSCAN
import sklearn
import sklearn.metrics
import pandas as pd
import time
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
# "arxiv-clustering-s2s": "arxiv",
#     "reddit-clustering": "reddit",
#     "imagenet": "ImageNet",
#     "mnist": "MNIST",
#     "birds": "birds",

dataset = "arxiv-clustering-s2s"
data = np.load(f"/home/sy/embeddings/{dataset}/{dataset}.npy").astype("float32")
labels = np.loadtxt(f"/home/sy/embeddings/{dataset}/{dataset}.gt").flatten()

# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# data, labels = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
# data = StandardScaler().fit_transform(data)


print(dataset, "data loaded")
#"mnist"
# for epsilon > 8, number cluster is 1.
## when epsilon =1, min sample > 2, everything is noise
# "arxiv-clustering-s2s"
eps_values = [0.4, 0.42, 0.44, 0.46,0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
min_samples_values = [1, 2, 3, 4, 5]

# Initialize an empty list to store results
results = []

# Iterate over parameter combinations
for eps in eps_values:
    for min_samples in min_samples_values:
        # Perform DBSCAN clustering
        start = time.time()
        dbscan_labels_, core_samples_mask = DBSCAN(data, eps=eps, min_samples=min_samples)
        
        # Compute evaluation metric (e.g., adjusted Rand index)
        ari = sklearn.metrics.adjusted_rand_score(labels, dbscan_labels_)
        
        # Append results to the list
        results.append({'dataset': dataset,
                        'eps': eps,
                        'min_samples': min_samples,
                        'num_clusters': len(np.unique(dbscan_labels_)),
                        'num_noise': np.sum(dbscan_labels_ == -1),
                        'ARI': ari})
        print(eps, min_samples, ari)

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Print or further analyze the results
print(results_df)
results_df.to_csv(f"../results/{dataset}.csv")