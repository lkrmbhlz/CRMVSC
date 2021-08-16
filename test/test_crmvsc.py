import pytest
from crmvsc.crmvsc import pairwise_co_regularization
from crmvsc.synthetic_data import synthetic_data_1

view_1, view_2, correct_labels = synthetic_data_1(1000)


def crmvsc_synthetic_data_1():
    assert view_1.shape[0] == 1000
    assert view_2.shape[0] == 1000
    assert correct_labels.shape[0] == 1000

    matrices, labels, nmi_scores = \
        pairwise_co_regularization(views=[view_1, view_2],
                                   correct_labels=correct_labels,
                                   number_of_clusters=2,
                                   number_of_iterations=10,
                                   lambda_value=0.025,
                                   views_as_similarity_matrices=False,
                                   laplacian_type='ng',
                                   random_state=42)

    assert nmi_scores[10][0] > 0.28
    assert nmi_scores[10][1] > 0.28

    assert nmi_scores[0][0] < nmi_scores[10][0]
    assert nmi_scores[0][1] < nmi_scores[10][1]

    return labels, nmi_scores


def test_if_crmvsc_synthetic_data_is_deterministic():
    last_nmi_scores = []
    for i in range(5):
        labels, nmi_scores = crmvsc_synthetic_data_1()
        if not len(last_nmi_scores) == 0:
            assert last_nmi_scores[-1][0] == nmi_scores[10][0]
            assert last_nmi_scores[-1][1] == nmi_scores[10][1]
        last_nmi_scores.append(nmi_scores[10])
