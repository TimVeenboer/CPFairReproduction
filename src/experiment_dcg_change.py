from copy import deepcopy
from datetime import datetime
import os
import pandas as pd

import numpy as np
from boxplot import create_boxplots
from clean_results import clean_results
from dataset_utils import read_item_groups, read_user_groups
from experiment import Experiment, _run_cornac_experiment, _write_experiment_results
from matrices import load_ground_truth_index, load_ranking_matrices, read_item_index
from optimisation import fairness_optimisation_dcg_change


class ExperimentDCG(Experiment):
    def __init__(self, config_path: str, models: list, metrics: list):
        super().__init__(config_path, models, metrics)
    
    def run_experiment(self):
        experiment_time_run = datetime.now().strftime('%d%m%Y%H%M%S')

        if not os.path.exists(os.getcwd() + '/results'):
            os.mkdir('results')
        os.mkdir('results/' + experiment_time_run)

        experiment_results = {}

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

                active_user_size = len(active_user_ids)
                inactive_user_size = len(inactive_user_ids)

                print(f"ActiveU: {active_user_size}, \
                      InActive: {inactive_user_size}, \
                        All: {active_user_size + inactive_user_size}")

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

                    shorthead_size = len(shorthead_item_ids)
                    longtail_size = len(longtail_item_ids)

                    print(f"No. of Shorthead Items: {shorthead_size} \
                          and No. of Longtaill Items: {longtail_size}")

                    for model in exp.models:
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
                                W, item_group = fairness_optimisation_dcg_change(
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
                                    W, item_group = fairness_optimisation_dcg_change(
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
                                    W, item_group = fairness_optimisation_dcg_change(
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
                                        W, item_group = fairness_optimisation_dcg_change(
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

                        if dataset in experiment_results:
                            experiment_results[dataset] = pd.concat([experiment_results[dataset], clean_results(results_df)])
                        else:
                            experiment_results[dataset] = clean_results(results_df)

            experiment_results[dataset].to_csv(
                f"results/{experiment_time_run}/results_{dataset}.csv", index=False)
            
            if self.config['boxplot']:
                create_boxplots(f"results/{experiment_time_run}/boxplots", dataset, experiment_results[dataset])
                        
        return experiment_results
