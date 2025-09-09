The first part of my Bachelor's Final Project is a deep dive trhought the state of the art of vector retrieval, i.e. solving k-NN (or; more precisely, A-NN; the aproximate version) as fast and accurate as possible, 
which is the core operation of RAG-based agents. To complement the theoretical explanations, I programmed all the algorithms and proccedures asocciated with them, divided in the following categories:

- Branch & Bound: Designed to divided the space and create a tree that covers all the subdivisions hierarchically.
- Locality Sensitive Hash: Maps the vectors with a hash function that maximizes collision between vectors proportionally to proximity, the closer the vectors are the higher the chance of returning the same value
- Graph algorithms: Establishes a graph that models conexions between vectors and perform the search over it.
- Clustering: Creates certain centroids with K-Means and perform the search first in the centroid space, to then search between the vectors associated with the selected cluster
- Quantization: Compress the search space reducing the number of vectors trying to preserve the most accuracy.
- Sketching: Reduce the dimension of the space throught several operations, trying to keep the proportionality between the distances
