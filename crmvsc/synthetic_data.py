import numpy as np
import pandas as pd


def synthetic_data_1(number_of_data_points_per_view: int) -> np.ndarray:
    """
    Generates a synthetic data set in manner of [#2]_.

    References
    ----------
    .. [#2] Abhishek Kumar, Piyush Rai, and Hal Daumé. 2011. Co-regularized multi-view spectral
        clustering. In Proceedings of the 24th International Conference on Neural Information Processing Systems (
        NIPS'11). Curran Associates Inc., Red Hook, NY, USA, 1413–1421.

    Parameters
    ----------
    number_of_data_points_per_view

    Returns
    -------

    view_1

    view_2

    correct_labels

    """

    n_clusters = 2

    means_view_1 = [(1, 1), (2, 2)]
    covariance_matrices_view_1 = [
        [[1, 0.5], [0.5, 1.5]],
        [[0.3, 0], [0, 0.6]]
    ]

    data_view_1_cluster_1 = np.random.multivariate_normal(means_view_1[0],
                                                          covariance_matrices_view_1[0],
                                                          int(number_of_data_points_per_view / n_clusters))

    data_view_1_cluster_2 = np.random.multivariate_normal(means_view_1[1],
                                                          covariance_matrices_view_1[1],
                                                          int(number_of_data_points_per_view / n_clusters))

    data = []
    for data_point in data_view_1_cluster_1:
        data.append([data_point[0], data_point[1], 0, 1])

    for data_point in data_view_1_cluster_2:
        data.append([data_point[0], data_point[1], 1, 1])

    means_view_2 = [(2, 2), (1, 1)]
    covariance_matrices_view_2 = [
        [[0.3, 0], [0, 0.6]],
        [[1, 0.5], [0.5, 1.5]]
    ]

    data_view_2_cluster_1 = np.random.multivariate_normal(means_view_2[0],
                                                          covariance_matrices_view_2[0],
                                                          int(number_of_data_points_per_view / n_clusters))

    data_view_2_cluster_2 = np.random.multivariate_normal(means_view_2[1],
                                                          covariance_matrices_view_2[1],
                                                          int(number_of_data_points_per_view / n_clusters))

    for data_point in data_view_2_cluster_1:
        data.append([data_point[0], data_point[1], 0, 2])

    for data_point in data_view_2_cluster_2:
        data.append([data_point[0], data_point[1], 1, 2])

    df = pd.DataFrame(data=data, columns=['x', 'y', 'cluster', 'view'])

    view_1 = df.loc[df['view'] == 1][['x', 'y']].values
    view_2 = df.loc[df['view'] == 2][['x', 'y']].values

    correct_labels = df.loc[df['view'] == 1]['cluster'].values

    return view_1, view_2, correct_labels
