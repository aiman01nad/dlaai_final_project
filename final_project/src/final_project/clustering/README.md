Steps in building the discrete codebook from the continous VAE latents:

1. Build a k-NN graph from the VAE latents.
    Using Annoy for approximate nearest neighbors using Euclidian distances. This graph serves as a discrete approximation of the latent manifold.

2. Choose landmark medoids.
    Selects landmark medoids using K-Medoids clustering with Euclidian distance as a scalable alternative to using geodesic distances.

3. Compute geodesic distances.
    Compute pairwise distances from each landmark using Dijkstra.

4. Assign each latent to the closest medoid.
    Using the shortest distances computed via shortest path in the k-NN graph. 
    Save codes for ar model training too, by reshaping the labels.

    Saves kmedoids_labels, shape (N * H * W, )
    Saves geodesic_codes, shape (N, H, W)

5. Build the codebook.
    Contains the medoid vectors chosen from the VAE latents.

    Saves codebook_latents, shape (num_codes, embedding_dim)