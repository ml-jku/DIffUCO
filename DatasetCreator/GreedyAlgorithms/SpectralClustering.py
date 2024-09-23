import igraph
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


def spectral_clustering(graph, num_clusters):
    # Get the adjacency matrix of the graph
    adjacency_matrix = np.array(graph.get_adjacency().data)

    # Laplacian matrix
    laplacian_matrix = np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix

    # Compute the eigenvectors and eigenvalues
    _, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Use the first 'num_clusters' eigenvectors as features
    features = eigenvectors[:, :num_clusters]

    # Perform k-means clustering on the features
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_assignments = kmeans.fit_predict(features)

    return cluster_assignments


if(__name__ == "__main__"):
    # Example usage
    # Create a sample graph
    g = igraph.Graph.Erdos_Renyi(n=20, m = 40)

    # Perform spectral clustering with 2 clusters
    num_clusters = 5
    clusters = spectral_clustering(g, num_clusters)

    # Print the cluster assignments
    print("Cluster Assignments:", clusters)
    print(len(clusters), len(set(clusters)))
    import networkx as nx
    from matplotlib import pyplot as plt
    g_nx = nx.from_edgelist(g.get_edgelist())

    labels = {node: cluster for node, cluster in zip(g_nx.nodes, clusters)}
    # Plot the graph and label nodes with cluster values
    pos = nx.spring_layout(g_nx)  # You can use other layout algorithms as well
    nx.draw(g_nx, pos, with_labels=True, labels=labels, cmap=plt.cm.rainbow)
    plt.title('Spectral Clustering of Graph')
    plt.show()
