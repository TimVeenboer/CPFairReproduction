import numpy as np
from tqdm.notebook import tqdm
from cornac.eval_methods import BaseMethod


def catalog_coverage(predicted: list, catalog: list) -> float:
    """
    Computes the catalog coverage for k lists of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    k: integer
        The number of observed recommendation lists
        which randomly choosed in our offline setup
    Returns
    ----------
    catalog_coverage:
        The catalog coverage of the recommendations as a percent rounded to 2 decimal places
    ----------
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    predicted_flattened = [p for sublist in predicted for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions / (len(catalog) * 1.0) * 100, 2)
    # output: precent (%)
    return catalog_coverage


def novelty(predicted: list, pop: dict, u: int, k: int) -> float:
    """
    Computes the novelty for a list of recommended items for a user
    Parameters
    ----------
    predicted : a list of recommedned items
        Ordered predictions
        example: ['X', 'Y', 'Z']
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    k: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    self_information = 0
    for item in predicted:
        if item in pop.keys():
            item_popularity = pop[item] / u
            item_novelty_value = np.sum(-np.log2(item_popularity))
        else:
            item_novelty_value = 0
        self_information += item_novelty_value
    novelty_score = self_information / k
    return novelty_score


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i, p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg


def metric_per_group(group: int, W: np.array, ground_truth, pop_items, P: np.array, eval_method: BaseMethod):
    NDCG10 = list()
    Pre10 = list()
    Rec10 = list()
    Novelty10 = list()
    predicted = list()
    All_Predicted = list()

    for uid in tqdm(group):
        if uid in ground_truth.keys():
            for j in range(50):
                if W[uid][j].x == 1:
                    predicted.append(P[uid][j])
            copy_predicted = predicted[:]
            All_Predicted.append(copy_predicted)
            NDCG = ndcgk(actual=ground_truth[uid], predicted=predicted)
            Pre = precisionk(actual=ground_truth[uid], predicted=predicted)
            Rec = recallk(actual=ground_truth[uid], predicted=predicted)
            Novelty = novelty(predicted=predicted, pop=pop_items,
                              u=eval_method.total_users, k=10)

            NDCG10.append(NDCG)
            Pre10.append(Pre)
            Rec10.append(Rec)
            Novelty10.append(Novelty)

            # cleaning the predicted list for a new user
            predicted.clear()

    catalog = catalog_coverage(
        predicted=All_Predicted, catalog=pop_items.keys())

    return round(np.mean(NDCG10), 5), round(np.mean(Pre10), 5), \
        round(np.mean(Rec10), 5), round(np.mean(Novelty10), 5), catalog


def metric_on_all(W: np.array, ground_truth, pop_items, P: np.array, eval_method: BaseMethod):
    """
    """
    predicted_user = list()
    NDCG_all = list()
    PRE_all = list()
    REC_all = list()
    Novelty_all = list()
    All_Predicted = list()

    for uid in tqdm(range(eval_method.total_users)):
        if uid in ground_truth.keys():
            for j in range(50):
                if W[uid][j].x == 1:
                    predicted_user.append(P[uid][j])

            copy_predicted = predicted_user[:]
            All_Predicted.append(copy_predicted)

            NDCG_user = ndcgk(
                actual=ground_truth[uid], predicted=predicted_user)
            PRE_user = precisionk(
                actual=ground_truth[uid], predicted=predicted_user)
            REC_user = recallk(
                actual=ground_truth[uid], predicted=predicted_user)
            Novelty_user = novelty(
                predicted=predicted_user, pop=pop_items, u=eval_method.total_users, k=10)

            NDCG_all.append(NDCG_user)
            PRE_all.append(PRE_user)
            REC_all.append(REC_user)
            Novelty_all.append(Novelty_user)

            # cleaning the predicted list for a new user
            predicted_user.clear()

    catalog = catalog_coverage(
        predicted=All_Predicted, catalog=pop_items.keys())

    return round(np.mean(NDCG_all), 5), round(np.mean(PRE_all), 5), round(np.mean(REC_all), 5), \
        round(np.mean(Novelty_all), 5), catalog
