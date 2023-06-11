import numpy as np
from cornac.eval_methods import BaseMethod
from mip import Model, xsum, maximize


def fairness_optimisation(
        fairness_mode,
        uepsilon,
        iepsilon,
        topk: int,
        eval_method: BaseMethod,
        no_item_groups: int,
        no_user_groups: int,
        S: np.array,
        U: np.array,
        Ihelp: np.array,
        Ahelp: np.array,
        train_checkins):
    print(
        f"Runing fairness optimisation on '{fairness_mode}', {uepsilon}, {iepsilon}")

    # V1: No. of users
    # V2: No. of top items (topk)
    # V3: No. of user groups
    # V4: no. og item groups
    V1, V2, V3, V4 = set(range(eval_method.total_users)), set(
        range(topk)), set(range(no_user_groups)), set(range(no_item_groups))

    # initiate model
    model = Model()

    # W is a matrix (size: user * top items) to be learned by model
    W = [[model.add_var() for j in V2] for i in V1]
    user_dcg = [model.add_var() for i in V1]
    user_ndcg = [model.add_var() for i in V1]
    group_ndcg_v = [model.add_var() for k in V3]
    item_group = [model.add_var() for k in V4]

    user_precision = [model.add_var() for i in V1]
    group_precision = [model.add_var() for k in V3]

    user_recall = [model.add_var() for i in V1]
    group_recall = [model.add_var() for k in V3]

    if fairness_mode == 'N':
        ### No Fairness ###
        model.objective = maximize(
            xsum((S[i][j] * W[i][j]) for i in V1 for j in V2))
    elif fairness_mode == 'C':
        ### C-Fairness: NDCG_Best: group_ndcg_v[1] - group_ndcg_v[0] ###
        model.objective = maximize(xsum(
            (S[i][j] * W[i][j]) for i in V1 for j in V2) - uepsilon * (group_ndcg_v[1] - group_ndcg_v[0]))
    elif fairness_mode == 'P':
        model.objective = maximize(xsum(
            (S[i][j] * W[i][j]) for i in V1 for j in V2) - iepsilon * (item_group[0] - item_group[1]))
    elif fairness_mode == 'CP':
        model.objective = maximize(xsum((S[i][j] * W[i][j]) for i in V1 for j in V2) - uepsilon * (
            group_ndcg_v[1] - group_ndcg_v[0]) - iepsilon * (item_group[0] - item_group[1]))

    # first constraint: the number of 1 in W should be equal to top-k, recommending top-k best items
    k = 10
    for i in V1:
        model += xsum(W[i][j] for j in V2) == k

    for i in V1:
        user_idcg_i = 7.137938133620551

        model += user_dcg[i] == xsum((W[i][j] * Ahelp[i][j]) for j in V2)
        model += user_ndcg[i] == user_dcg[i] / user_idcg_i

        model += user_precision[i] == xsum((W[i][j] * Ahelp[i][j])
                                           for j in V2) / k
        model += user_recall[i] == xsum((W[i][j] * Ahelp[i][j])
                                        for j in V2) / len(train_checkins[i])

    for k in V3:
        model += group_ndcg_v[k] == xsum(user_dcg[i] * U[i][k] for i in V1)
        model += group_precision[k] == xsum(user_precision[i]
                                            * U[i][k] for i in V1)
        model += group_recall[k] == xsum(user_recall[i] * U[i][k] for i in V1)

    for k in V4:
        model += item_group[k] == xsum(W[i][j] * Ihelp[i][j][k]
                                       for i in V1 for j in V2)

    for i in V1:
        for j in V2:
            model += W[i][j] <= 1
    # optimizing
    model.optimize()

    return W, item_group