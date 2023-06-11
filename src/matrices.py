import numpy as np
from tqdm import tqdm


def load_ranking_matrices(model, total_users, total_items, topk):
    # S is a matrix to store user's scores on each item
    # P includes the indices of topk ranked items
    # Sprime saves the scores of topk ranked items
    S = np.zeros((total_users, total_items))
    P = np.zeros((total_users, topk))

    # for model in exp.models:
    print(model.name)
    for uid in tqdm(range(total_users)):
        S[uid] = model.score(uid)
        P[uid] = np.array(list(reversed(model.score(uid).argsort()))[:topk])

    return S, P


def load_ground_truth_index(total_users: int, topk: int, P: np.array, train_checkins):
    # Ahelp is a binary matrix in which an element of its is 1 if the corresponding element in P (which is an item index) is in ground truth.
    # Actually is shows whether the rankied item in P is included in ground truth or not.
    Ahelp = np.zeros((total_users, topk))
    for uid in tqdm(range(total_users)):
        for j in range(topk):
            # convert user_ids to user_idx
            # convert item_ids to item_idx
            if P[uid][j] in train_checkins[uid]:
                Ahelp[uid][j] = 1
    return Ahelp


def read_item_index(total_users: int, topk: int, no_item_groups: int, P: np.array,
                    shorthead_item_ids: set, longtail_item_ids: set):
    Ihelp = np.zeros((total_users, topk, no_item_groups))
    for uid in range(total_users):
        for lid in range(topk):
            # convert item_ids to item_idx
            if P[uid][lid] in shorthead_item_ids:
                Ihelp[uid][lid][0] = 1
            elif P[uid][lid] in longtail_item_ids:
                Ihelp[uid][lid][1] = 1
    return Ihelp
