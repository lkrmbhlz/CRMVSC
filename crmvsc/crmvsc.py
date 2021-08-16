import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import k_means
from sklearn.metrics import normalized_mutual_info_score


def pairwise_co_regularization(views: np.ndarray,
                               correct_labels : np.ndarray,
                               number_of_clusters: int,
                               number_of_iterations: int,
                               lambda_value: float = 0.025,
                               views_as_similarity_matrices: bool = False,
                               laplacian_type: str = 'sym',
                               random_state=None):
    """
    Co-Regularized Multi-view Spectral Clustering algorithm by Kumar, Rai, and Daumé.[#2]_

    References
    ----------
    .. [#1] Abhishek Kumar, Piyush Rai, and Hal Daumé. 2011. Co-regularized multi-view spectral clustering. In Proceedings of the 24th International Conference on Neural Information Processing Systems (NIPS'11). Curran Associates Inc., Red Hook, NY, USA, 1413–1421.

    Parameters
    ----------
    views : array-like, shape (number of data objects, number of features) or shape (number of data objects, number of data objects) if provided as precalculated similarity matrices
        The views (data) as list of numpy arrays, must be of same size.

    correct_labels : array-like, shape (number of data objects, 1)

    number_of_clusters : int

    number_of_iterations : int
        Usually less than 10 iterations necessary until convergence.

    lambda_value : float
        Hyperparameter, trades-off spectral clustering objectives and spectral embedding (dis)agreement term.
        Usually small (< 0.03).

    views_as_similarity_matrices : bool
         True if the views are not passed as data points but already as similiarity matrices.

    laplacian_type : str
        The type of Laplacian to be used ('ng', 'sym', or 'rw').

    random_state : int, RandomState instance or None
        Random state for k-means step.

    Returns
    -------

    matrices : dict
        A dictionary containing the eigenvector matrices U, laplacian matrices L and the similarity matrices of U KU
        for the initial state (0) and each iteration.

    labels : dict
         A dictionary containing the labels assigned by k-means for the initial state (0) and at each iteration.

    nmi_scores : dict
        A dictionary containing the NMI score for each view for the initial state (0) and at each iteration.
    """

    if not [np.array(view).shape == np.array(view).shape for view in views]:
        raise Exception('Views must be of same shape.')

    matrices = {0: {}}
    matrices[0]['L'] = {}
    matrices[0]['U'] = {}

    L = {}
    U = {}

    for i, view in enumerate(views):

        if views_as_similarity_matrices:
            sim_matrix = view
        else:
            sim_matrix = get_affinity_matrix(view)
            print('Calculated similarity matrix of view ', i)

        if laplacian_type == 'sym':
            normalized_laplacian = symmetric_normalized_laplacian(sim_matrix)
            print('Calculated symmetric Laplacian ', i)
        if laplacian_type == 'rw':
            normalized_laplacian = random_walk_normalized_laplacian(sim_matrix)
            print('Calculated random walk Laplacian ', i)
        if laplacian_type == 'ng':
            normalized_laplacian = ng_normalized_laplacian(sim_matrix)
            print('Calculated Ng Laplacian ', i)

        L[i] = normalized_laplacian
        matrices[0]['L'][i] = L[i]

        if laplacian_type == 'ng':
            U[i] = get_k_largest_eigenvectors(normalized_laplacian, number_of_clusters)
        else:
            U[i] = get_k_smallest_eigenvectors(normalized_laplacian, number_of_clusters)

        matrices[0]['U'][i] = U[i]

    nmi_scores = {0: {}}

    labels = {0: {}}

    for view, u in U.items():
        _, view_labels, _ = k_means(u, number_of_clusters, random_state=random_state)
        labels[0][view] = view_labels
        nmi_scores[0][view] = normalized_mutual_info_score(view_labels, correct_labels)
    print('Initial NMI: ', nmi_scores[0])

    for i in range(number_of_iterations):
        matrices[i+1] = {}
        matrices[i+1]['L'] = {}
        matrices[i+1]['U'] = {}
        matrices[i+1]['KU'] = {}

        labels[i + 1] = {}

        nmi_scores[i + 1] = {}

        for j, view in enumerate(views):
            KU = sum(np.matmul(eigenvectors, eigenvectors.T) for k, eigenvectors in U.items() if k != j)
            lap = L[j] + lambda_value * KU

            if laplacian_type == 'ng':
                U[j] = get_k_largest_eigenvectors(lap, number_of_clusters)
            else:
                U[j] = get_k_smallest_eigenvectors(lap, number_of_clusters)
            L[j] = lap

            matrices[i+1]['L'][j] = L[j]
            matrices[i+1]['U'][j] = U[j]
            matrices[i+1]['KU'][j] = KU

        for view, u in U.items():
            _, view_labels, _ = k_means(u, number_of_clusters, random_state=random_state)
            labels[i + 1][view] = view_labels
            nmi_scores[i + 1][view] = normalized_mutual_info_score(view_labels, correct_labels)

        print('Iteration: ', i+1)
        print("NMI scores: ", nmi_scores[i+1])

        # Measure of disagreement (to be minimized)
        if not len(views) == 1:
            print("Disagreement: ",
                  - np.linalg.multi_dot(
                      [np.matmul(eigenvectors, eigenvectors.T) for m, eigenvectors in U.items()]).trace())

    return matrices, labels, nmi_scores


def ng_normalized_laplacian(weight_matrix):
    """
    The normalized Laplacian matrix of a graph introduced by Ng et al. in [#1]_.:
    (D^-1/2)W(D^-1/2) (D: degree matrix, W: weight matrix)

    Explanation from the paper:
    "As replacing L with I-L would complicate our later discussion and only changes the eigenvalues
    (from lambda_i to 1 - lambda_i) and not the eigenvectors, we instead use L"

    Parameters
    ----------
    weight_matrix : array-like, shape (number of nodes, number of nodes)
        The weight matrix or adjacency matrix of a graph.

    References
    ----------
    .. [#1] Ng, A.Y. & Jordan, Michael & Weiss, Y. (2001). On Spectral Clustering: Analysis and an Algorithm. Adv. Neural Inf. Process. Syst.. 2.

    Returns
    -------
    ng_laplacian: array-like, shape (number of nodes, number of nodes)
        the normalized Laplacian
    """

    # Degree of a node: sum over all adjacent weights.
    degrees = np.sum(weight_matrix, axis=1)

    inverted_degrees = [1 / degree if degree != 0 else 0 for degree in degrees]
    sqrt_inverted_degree_matrix = np.sqrt(inverted_degrees * np.eye(len(degrees)))

    ng_laplacian = np.linalg.multi_dot(
        [sqrt_inverted_degree_matrix, weight_matrix, sqrt_inverted_degree_matrix])

    return ng_laplacian


def symmetric_normalized_laplacian(weight_matrix):
    """
    The symmetric normalized Laplacian matrix of a graph:
    I - (D^-1/2)W(D^-1/2) (D: degree matrix, W: weight matrix)

    (See https://en.wikipedia.org//wiki/Laplacian_matrix)

    Parameters
    ----------
    weight_matrix : array-like, shape (number of nodes, number of nodes)
        The weight matrix or adjacency matrix of a graph.

    Returns
    -------
    symmetric_laplacian: array-like, shape (number of nodes, number of nodes)
        the symmetric normalized Laplacian matrix
    """

    return np.eye(len(weight_matrix)) - ng_normalized_laplacian(weight_matrix)


def random_walk_normalized_laplacian(weight_matrix):
    """
    The random walk normalized Laplacian matrix of a graph:
    I - (D^-1)W (D: degree matrix, W: weight matrix)

    (See https://en.wikipedia.org//wiki/Laplacian_matrix)

    Parameters
    ----------
    weight_matrix : array-like, shape (number of nodes, number of nodes)
        The weight matrix or adjacency matrix of a graph.

    Returns
    -------
    random_walk_laplacian: array-like, shape (number of nodes, number of nodes)
    """

    # Degree of a node: sum over all adjacent weights.
    degrees = np.sum(weight_matrix, axis=1)

    inverted_degrees = [1 / degree if degree != 0 else 0 for degree in degrees]
    inverted_degree_matrix = inverted_degrees * np.eye(len(degrees))

    return np.eye(len(degrees)) - np.matmul(inverted_degree_matrix, weight_matrix)


def get_k_largest_eigenvectors(matrix, k):
    """
    Returns the k largest eigenvectors of a matrix (the eigenvectors corresponding to the k largest eigenvalues).

    Parameters
    ----------
    matrix : array-like, shape (n, n)

    k: int

    Returns
    -------
    eigenvectors: array-like, shape (n, k)
    """

    n = len(matrix)
    eigenvalues, eigenvectors = eigh(matrix, subset_by_index=[n - k, n - 1])

    return eigenvectors


def get_k_smallest_eigenvectors(matrix, k):
    """
    Returns the k smallest eigenvectors of a matrix (the eigenvectors corresponding to the k smallest eigenvalues).

    Parameters
    ----------
    matrix : array-like, shape (n, n)

    k: int

    Returns
    -------
    k_smallest_eigenvectors: array-like, shape (n, k)
    """

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort the eigenvalues by their L2 norms and record the indices
    indices = np.argsort(np.linalg.norm(np.reshape(eigenvalues, (1, len(eigenvalues))), axis=0))

    k_smallest_eigenvectors = np.real(eigenvectors[:, indices[:k]])

    # Flip signs of an eigenvector (column) if the first element of the eigenvector is negative
    for i, column in enumerate(k_smallest_eigenvectors.T):
        if np.sign(column[0]) == 1.0:
            k_smallest_eigenvectors.T[i] = column * (-1)

    return k_smallest_eigenvectors


def adjacency_matrix(dataset: np.ndarray, metric: str) -> np.ndarray:
    """
    Calculates the pairwise distances of the row vectors of X. Returns an adjacency or distance matrix.

    Parameters
    ----------
    dataset : array-like, shape (number_of_data_objects, number_of_features)

    metric: str
        String value of a metric. Examples: ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’.

    Returns
    -------
    adjacency_matrix: array-like, shape (number_of_data_objects, number_of_data_objects)
    """

    return squareform(pdist(dataset, metric=metric))


def get_affinity_matrix(dataset, sigma=None):
    """
    Calculates euclidean adjacency matrix from a dataset and transforms the resulting distance matrix for which 0 means
    identical  elements, and high values means very dissimilar elements, into an affinity matrix by applying the
    Gaussian (RBF, heat) kernel: A = exp(-||s_i - s_j||^2 / (2 * sigma)) if i != j and A_ii = O. (A: adjacency matrix)
    [#1]_

    If no sigma is given, we simply take the median of the adjacency matrix as sigma as proposed in [#2]_.


    References
    ----------
    .. [#1] Ng, A.Y. & Jordan, Michael & Weiss, Y. (2001). On Spectral Clustering: Analysis and an Algorithm. Adv. Neural Inf. Process. Syst.. 2.
    .. [#2] Abhishek Kumar, Piyush Rai, and Hal Daumé. 2011. Co-regularized multi-view spectral clustering. In Proceedings of the 24th International Conference on Neural Information Processing Systems (NIPS'11). Curran Associates Inc., Red Hook, NY, USA, 1413–1421.


    Parameters
    ----------
    dataset : array-like, shape (number_of_data_objects, number_of_features)

    sigma: float
        Free parameter representing the width of the Gaussian kernel.

    Returns
    -------
    affinity_matrix: array-like, shape (number_of_data_objects, number_of_data_objects)
        Ranges from 0 (no similarity) to 1 (maximum similarity).
    """

    adjacency = adjacency_matrix(dataset, metric='euclidean')

    if sigma is None:
        sigma = np.median(adjacency)

    affinity_matrix = np.exp(- adjacency / (2 * sigma))
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix
