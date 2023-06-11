from copy import deepcopy
from datetime import datetime
import os
import yaml
import cornac
from cornac.eval_methods import BaseMethod
import pandas as pd

from clean_results import clean_results
from dataset_utils import *
from matrices import *
from metrics import metric_per_group, metric_on_all
from optimisation import fairness_optimisation


def _write_experiment_results(
        results_df: pd.DataFrame,
        fair_mode: str,
        W: np.array,
        active_user_ids: set,
        inactive_user_ids: set,
        dataset: str,
        model_name: str,
        u_group: int,
        i_group: int,
        user_eps: float,
        item_eps: float,
        eval_method: BaseMethod,
        item_group,
        ground_truth,
        pop_items,
        P):

    # Calculate the metrics for both groups and all users/items
    ndcg_ac, pre_ac, rec_ac, novelty_ac, coverage_ac = metric_per_group(
        group=active_user_ids, W=W, ground_truth=ground_truth, pop_items=pop_items, P=P, eval_method=eval_method)
    ndcg_iac, pre_iac, rec_iac, novelty_iac, coverage_iac = metric_per_group(
        group=inactive_user_ids, W=W, ground_truth=ground_truth, pop_items=pop_items, P=P, eval_method=eval_method)
    ndcg_all, pre_all, rec_all, novelty_all, coverage_all = metric_on_all(
        W=W, ground_truth=ground_truth, pop_items=pop_items, P=P, eval_method=eval_method)

    if user_eps is not None:
        user_eps_string = format(user_eps, '.7f')
    else:
        user_eps_string = '-'

    if item_eps is not None:
        item_eps_string = format(item_eps, '.7f')
    else:
        item_eps_string = '-'

    results = [dataset, model_name, u_group, i_group, fair_mode, user_eps_string, item_eps_string,
               ndcg_all, ndcg_ac, ndcg_iac, pre_all, pre_ac, pre_iac, rec_all, rec_ac, rec_iac,
               novelty_all, novelty_ac, novelty_iac, coverage_all, coverage_ac, coverage_iac,
               item_group[0].x, item_group[1].x, f"{eval_method.total_users*10}=={item_group[0].x + item_group[1].x}"]
    results_df.loc[len(results_df)] = results

    return results_df


def _run_cornac_experiment(dataset: str, models: list, metrics: list):
    print(f"Datasets: {dataset}")
    # read train, tune, test datasets
    train_data, _, test_data = read_data(dataset=dataset)
    # load data into Cornac and create eval_method
    eval_method = BaseMethod.from_splits(
        train_data=train_data,
        test_data=test_data,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=True
    )

    total_users = eval_method.total_users
    total_items = eval_method.total_items
    # load train_checkins and pop_items dictionary
    train_checkins, pop_items = read_train_data(
        os.getcwd() + f"/datasets/{dataset}/{dataset}_train.txt", eval_method=eval_method)
    # load ground truth dict
    ground_truth = read_ground_truth(f"datasets/{dataset}/{dataset}_test.txt", eval_method=eval_method)

    # run Cornac models and create experiment object including models' results
    exp = cornac.Experiment(eval_method=eval_method, models=models, metrics=metrics)
    exp.run()

    return eval_method, total_users, total_items, train_checkins, pop_items, ground_truth, exp


class Experiment():
    def __init__(self, config_path: str, models: list, metrics: list):
        if not os.path.exists(config_path):
            raise ValueError(
                "The path to the config file of the experiment was invalid!")

        self.models = models
        self.metrics = metrics

        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.fairness_categories = self.config['fairness_categories']
        self.download_data()

    def download_data(self):
        # Download all the datasets in the configuration and create their user and item groups.
        download_datasets(self.config['ds_names'])
        download_user_groups(
            self.config['ds_names'], self.config['ds_user_groups'])
        download_item_groups(
            self.config['ds_names'], self.config['ds_item_groups'])

    def run_experiment(self):
        experiment_time_run = datetime.now().strftime('%d%m%Y%H%M%S')

        if not os.path.exists(os.getcwd() + '/results'):
            os.mkdir('results')
        os.mkdir('results/' + experiment_time_run)

        for dataset in self.config['ds_names']:
            eval_method, total_users, total_items, \
                train_checkins, pop_items, ground_truth, exp = _run_cornac_experiment(
                    dataset, deepcopy(self.models), self.metrics)

            for user_group in self.config['ds_user_groups']:
                # read matrix U for users and their groups
                U = np.zeros((total_users, self.config['no_of_user_groups']))

                # load active and inactive users
                active_user_ids = read_user_groups(
                    user_group_fpath=os.getcwd() + f"/user_groups/{dataset}/{user_group}/active_ids.txt", gid=0,
                    U=U, eval_method=eval_method)
                inactive_user_ids = read_user_groups(
                    user_group_fpath=os.getcwd() + f"/user_groups/{dataset}/{user_group}/inactive_ids.txt", gid=1,
                    U=U, eval_method=eval_method)

                print(f"ActiveU: {len(active_user_ids)}, \
                      InActive: {len(inactive_user_ids)}, \
                        All: {len(active_user_ids) + len(inactive_user_ids)}")

                for i_group in self.config['ds_item_groups']:
                    # read matrix I for items and their groups
                    I = np.zeros(
                        (total_items, self.config['no_of_item_groups']))

                    # read item groups
                    shorthead_item_ids = read_item_groups(
                        item_group_fpath=os.getcwd() + f"/item_groups/{dataset}/{i_group}/shorthead_items.txt", gid=0,
                        eval_method=eval_method, I=I)
                    longtail_item_ids = read_item_groups(
                        item_group_fpath=os.getcwd() + f"/item_groups/{dataset}/{i_group}/longtail_items.txt", gid=1,
                        eval_method=eval_method, I=I)

                    print(f"No. of Shorthead Items: {len(shorthead_item_ids)} \
                          and No. of Longtaill Items: {len(longtail_item_ids)}")

                    for model in exp.models:
                        # results = open(
                        #     f"results/{experiment_time_run}/results_{dataset}_{model.name}.csv", 'w')
                        # results.write("Dataset,Model,GUser,GItem,Type,User_EPS,Item_EPS,ndcg_ALL,ndcg_ACT,ndcg_INACT,Pre_ALL,Pre_ACT,Pre_INACT,Rec_ALL,Rec_ACT,Rec_INACT,Nov_ALL,Nov_ACT,Nov_INACT,Cov_ALL,Cov_ACT,Cov_INACT,Short_Items,Long_Items,All_Items\n")
                        results_df = pd.DataFrame(columns=[
                            "Dataset", "Model", "GUser", "GItem", "Type", "User_EPS", "Item_EPS",
                            "ndcg_ALL", "ndcg_ACT", "ndcg_INACT", "Pre_ALL", "Pre_ACT", "Pre_INACT",
                            "Rec_ALL", "Rec_ACT", "Rec_INACT", "Nov_ALL", "Nov_ACT", "Nov_INACT",
                            "Cov_ALL", "Cov_ACT", "Cov_INACT", "Short_Items", "Long_Items", "All_Items"
                        ])

                        print(f"> Model: {model.name}")
                        # load matrix S and P
                        S, P = load_ranking_matrices(model=model, total_users=total_users,
                                                     total_items=total_items, topk=self.config['topk'])

                        # load matrix Ahelp
                        Ahelp = load_ground_truth_index(total_users=total_users, topk=self.config['topk'],
                                                        P=P, train_checkins=train_checkins)

                        # load matrix Ihelp
                        Ihelp = read_item_index(total_users=total_users, topk=50,
                                                no_item_groups=self.config['no_of_item_groups'],
                                                P=P, shorthead_item_ids=shorthead_item_ids,
                                                longtail_item_ids=longtail_item_ids)

                        # iterate on fairness mode: user, item, user-item
                        for fair_mode in self.config['fairness_categories']:
                            if fair_mode == 'N':
                                W, item_group = fairness_optimisation(
                                    fairness_mode=fair_mode,
                                    uepsilon=None,
                                    iepsilon=None,
                                    topk=self.config['topk'],
                                    eval_method=eval_method,
                                    no_item_groups=self.config['no_of_item_groups'],
                                    no_user_groups=self.config['no_of_user_groups'],
                                    S=S,
                                    U=U,
                                    Ihelp=Ihelp,
                                    Ahelp=Ahelp,
                                    train_checkins=train_checkins)

                                _write_experiment_results(
                                    results_df=results_df,
                                    fair_mode=fair_mode,
                                    W=W,
                                    active_user_ids=active_user_ids,
                                    inactive_user_ids=inactive_user_ids,
                                    dataset=dataset,
                                    model_name=model.name,
                                    u_group=user_group,
                                    i_group=i_group,
                                    user_eps=None,
                                    item_eps=None,
                                    eval_method=eval_method,
                                    item_group=item_group,
                                    ground_truth=ground_truth,
                                    pop_items=pop_items,
                                    P=P
                                )

                            if fair_mode == 'C':
                                for user_eps in self.config['user_epsilon']:
                                    W, item_group = fairness_optimisation(
                                        fairness_mode=fair_mode,
                                        uepsilon=user_eps,
                                        iepsilon=None,
                                        topk=self.config['topk'],
                                        eval_method=eval_method,
                                        no_item_groups=self.config['no_of_item_groups'],
                                        no_user_groups=self.config['no_of_user_groups'],
                                        S=S,
                                        U=U,
                                        Ihelp=Ihelp,
                                        Ahelp=Ahelp,
                                        train_checkins=train_checkins)

                                    _write_experiment_results(
                                        results_df=results_df,
                                        fair_mode=fair_mode,
                                        W=W,
                                        active_user_ids=active_user_ids,
                                        inactive_user_ids=inactive_user_ids,
                                        dataset=dataset,
                                        model_name=model.name,
                                        u_group=user_group,
                                        i_group=i_group,
                                        user_eps=user_eps,
                                        item_eps=None,
                                        eval_method=eval_method,
                                        item_group=item_group,
                                        ground_truth=ground_truth,
                                        pop_items=pop_items,
                                        P=P
                                    )

                            if fair_mode == 'P':
                                for item_eps in self.config['item_epsilon']:
                                    W, item_group = fairness_optimisation(
                                        fairness_mode=fair_mode,
                                        uepsilon=None,
                                        iepsilon=item_eps,
                                        topk=self.config['topk'],
                                        eval_method=eval_method,
                                        no_item_groups=self.config['no_of_item_groups'],
                                        no_user_groups=self.config['no_of_user_groups'],
                                        S=S,
                                        U=U,
                                        Ihelp=Ihelp,
                                        Ahelp=Ahelp,
                                        train_checkins=train_checkins)

                                    _write_experiment_results(
                                        results_df=results_df,
                                        fair_mode=fair_mode,
                                        W=W,
                                        active_user_ids=active_user_ids,
                                        inactive_user_ids=inactive_user_ids,
                                        dataset=dataset,
                                        model_name=model.name,
                                        u_group=user_group,
                                        i_group=i_group,
                                        user_eps=None,
                                        item_eps=item_eps,
                                        eval_method=eval_method,
                                        item_group=item_group,
                                        ground_truth=ground_truth,
                                        pop_items=pop_items,
                                        P=P
                                    )

                            if fair_mode == 'CP':
                                for user_eps in self.config['user_epsilon']:
                                    for item_eps in self.config['item_epsilon']:
                                        W, item_group = fairness_optimisation(
                                            fairness_mode=fair_mode,
                                            uepsilon=user_eps,
                                            iepsilon=item_eps,
                                            topk=self.config['topk'],
                                            eval_method=eval_method,
                                            no_item_groups=self.config['no_of_item_groups'],
                                            no_user_groups=self.config['no_of_user_groups'],
                                            S=S,
                                            U=U,
                                            Ihelp=Ihelp,
                                            Ahelp=Ahelp,
                                            train_checkins=train_checkins)

                                        _write_experiment_results(
                                            results_df=results_df,
                                            fair_mode=fair_mode,
                                            W=W,
                                            active_user_ids=active_user_ids,
                                            inactive_user_ids=inactive_user_ids,
                                            dataset=dataset,
                                            model_name=model.name,
                                            u_group=user_group,
                                            i_group=i_group,
                                            user_eps=user_eps,
                                            item_eps=item_eps,
                                            eval_method=eval_method,
                                            item_group=item_group,
                                            ground_truth=ground_truth,
                                            pop_items=pop_items,
                                            P=P
                                        )

                        results_df = clean_results(results_df)
                        results_df.to_csv(
                            f"results/{experiment_time_run}/results_{dataset}_{model.name}.csv", index=False)