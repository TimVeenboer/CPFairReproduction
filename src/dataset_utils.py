import os
from collections import defaultdict
from cornac.eval_methods import BaseMethod

from cornac.data import Reader
import numpy as np


def download_datasets(ds_names: list):
    ds_root_path = "datasets/"
    for dataset in ds_names:
        dataset_path = os.path.join(ds_root_path, dataset)

        if not os.path.isdir(dataset_path):
            os.makedirs(dataset_path)
            print("Directory '%s' is created." % dataset_path)
        else:
            print("Directory '%s' is exist." % dataset_path)

        # -nc: skip downloads that would download to existing files.

        try:
            os.system(
                f"wget -P {dataset_path} -nc https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/{dataset}_train.txt")
            os.system(
                f"wget -P {dataset_path} -nc https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/{dataset}_test.txt")
            os.system(
                f"wget -P {dataset_path} -nc https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/{dataset}_tune.txt")
            print(f"{dataset}: The train, tune, and test sets downloaded.")
        except Exception as e:
            print(e)


def download_user_groups(ds_names: list, ds_users: list):
    user_root_path = "user_groups/"
    for dataset in ds_names:
        for ugroup in ds_users:
            user_groups_path = os.path.join(user_root_path, dataset, ugroup)

            if not os.path.isdir(user_groups_path):
                os.makedirs(user_groups_path)
                print("Directory '%s' is created." % user_groups_path)
            else:
                print("Directory '%s' is exist." % user_groups_path)

            # -nc: skip downloads that would download to existing files.

            try:
                os.system(
                    f"wget -P {user_groups_path} https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/groups/users/{ugroup}/active_ids.txt")
                os.system(
                    f"wget -P {user_groups_path} https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/groups/users/{ugroup}/inactive_ids.txt")
                print(f"{dataset}: User groups on '{ugroup}' downloaded.")
            except Exception as e:
                print(e)


def download_item_groups(ds_names: list, ds_items: list):
    item_root_path = "item_groups/"
    for dataset in ds_names:
        for igroup in ds_items:
            item_groups_path = os.path.join(item_root_path, dataset, igroup)

            if not os.path.isdir(item_groups_path):
                os.makedirs(item_groups_path)
                print("Directory '%s' is created." % item_groups_path)
            else:
                print("Directory '%s' is exist." % item_groups_path)

            # -nc: skip downloads that would download to existing files.
        try:
            os.system(
                f"wget -P {item_groups_path} -nc https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/groups/items/{igroup}/shorthead_items.txt")
            os.system(
                f"wget -P {item_groups_path} -nc https://raw.githubusercontent.com/rahmanidashti/CPFairRecSys/main/datasets/{dataset}/groups/items/{igroup}/longtail_items.txt")
            print(f"{dataset}: Item groups on '{igroup}' downloaded.")
        except Exception as e:
            print(e)


def read_data(dataset):
    """
    Read the train, test, and tune file using Cornac reader class

    Parameters
    ----------
    dataset : the name of the dataset
      example: 'MovieLens100K'

    Returns
    ----------
    train_data:
      The train set that is 70% of interactions
    tune_data:
      The tune set that is 10% of interactions
    test_data:
      The test set that is 20% of interactions
    """
    reader = Reader()
    train_data = reader.read(
        fpath=os.getcwd() + f"/datasets/{dataset}/{dataset}_train.txt", fmt='UIR', sep='\t')
    tune_data = reader.read(
        fpath=os.getcwd() + f"/datasets/{dataset}/{dataset}_tune.txt", fmt='UIR', sep='\t')
    test_data = reader.read(
        fpath=os.getcwd() + f"/datasets/{dataset}/{dataset}_test.txt", fmt='UIR', sep='\t')
    return train_data, tune_data, test_data


def read_user_groups(user_group_fpath: str, gid: int, U, eval_method: BaseMethod) -> set:
    """
    Read the user groups lists

    Parameters
    ----------
    user_group_fpath:
      The path of the user group file

    U (global variabvle):
      The global matrix of users and their group

    Returns
    ----------
    user_ids:
      The set of user ids corresponding to the group
    """

    user_group = open(user_group_fpath, 'r').readlines()
    user_ids = set()
    for eachline in user_group:
        uid = eachline.strip()
        # convert uids to uidx
        uid = eval_method.train_set.uid_map[uid]
        uid = int(uid)
        user_ids.add(uid)
        U[uid][gid] = 1
    return user_ids


def read_item_groups(item_group_fpath: str, gid: int, eval_method: BaseMethod, I) -> set:
    item_group = open(item_group_fpath, 'r').readlines()
    item_ids = set()
    for eachline in item_group:
        iid = eachline.strip()
        # convert iids to iidx
        iid = eval_method.train_set.iid_map[iid]
        iid = int(iid)
        item_ids.add(iid)
        I[iid][gid] = 1
    return item_ids


def read_ground_truth(test_file: str, eval_method: BaseMethod):
    """
    Read test set data

    Parameters
    ----------
    test_file:
      The test set data

    Returns
    ----------
    ground_truth:
      A dictionary includes user with actual items in test data
    """
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, iid, _ = eachline.strip().split()

        # convert uids to uidx
        uid = eval_method.train_set.uid_map[uid]
        # convert iids to iidx
        iid = eval_method.train_set.iid_map[iid]

        uid, iid = int(uid), int(iid)
        ground_truth[uid].add(iid)
    return ground_truth


def read_train_data(train_file: str, eval_method: BaseMethod):
    """
    Read test set data

    Parameters
    ----------
    train_file:
      The train_file set data

    Returns
    ----------
    train_checkins:
      A dictionary includes user with items in train data
    pop: dictionary
      A dictionary of all items alongside of its occurrences counter in the training data
      example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    """
    train_checkins = defaultdict(set)
    pop_items = dict()
    train_data = open(train_file, 'r').readlines()

    for eachline in train_data:
        uid, iid, _ = eachline.strip().split()

        # convert uids to uidx
        uid = eval_method.train_set.uid_map[uid]
        # convert iids to iidx
        iid = eval_method.train_set.iid_map[iid]

        uid, iid = int(uid), int(iid)
        # a dictionary of popularity of items
        if iid in pop_items.keys():
            pop_items[iid] += 1
        else:
            pop_items[iid] = 1
        train_checkins[uid].add(iid)
    return train_checkins, pop_items
