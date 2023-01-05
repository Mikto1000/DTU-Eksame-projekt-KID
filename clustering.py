import numpy as np

def k_means(data, k, max_iter=100):
    # Initialize the centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        # Compute the distance from each point to each centroid
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=-1))
        
        # Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)
        
        # Compute the new centroids as the mean of the points assigned to each centroid
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check if the centroids have changed
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids


# Testing 1-, 2- and 3-dimensional data
for i in range(1,4):
    data = np.random.rand(50, i)
    data = np.append(data, data+1, axis=0)
    cluster_centers = k_means(data, 2)
    print(cluster_centers)

