import os
import pandas as pd
import numpy as np
from rliable import library as rly
from rliable import metrics


max_score_dict = {
    "give_way": 8.51
    # Add more environments and their corresponding random rewards and max scores here
}

random_reward_dict = {"give_way": 0.0}

def preprocess(env_name, algo_name) -> dict:
    """
    Each data: csv files with columns: step, seeds (10 seeds)
    """
    
    data_folder = os.path.join(f"./data/{env_name}/{Metric}/{algo_name}.csv")
    if not os.path.exists(data_folder):
        return None
    df = pd.read_csv(data_folder)
    df = df[[col for col in df.columns if "__MAX" not in col and "__MIN" not in col]]
    final_scores = []
    steps = df['Step'].values
    max_step = steps[-1]
    min_step = steps[0]
    total_range = max_step - min_step
    for col in df.columns:
        if col != "Step":
            seed_values = df[col].values
            normalized_values = (seed_values - random_reward_dict[env_name]) / (max_score_dict[env_name] - random_reward_dict[env_name])
            auc = np.trapezoid(normalized_values, x=steps) / total_range
            final_scores.append(auc)
    np_scores = np.array([final_scores]).T  
    return np_scores
def main():
    envs = ["give_way"] 
    algos = [
        "mappo",
        "sema",
        "l2m2",
        "l2m2_wo_subreward"
    ]

    print("=" * 100)
    print(f"{'ENV':<10} | {'ALGORITHM RANKING (Descending by IQM Score)':<90}")
    print("-" * 100)
    for env in envs:
        algo_scores = {}
        for algo in algos:
            np_scores = preprocess(env, algo)
            if np_scores is not None:
                # algo_scores[algo] = np_scores
                mean_score = np.mean(np_scores)
                std_score = np.std(np_scores)
                algo_scores[algo] = (mean_score, std_score)
            else:
                print(f"Warning: Data for {algo} in {env} not found.")
        # aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x)])
        # # print(algo_scores)
        # iqm_scores, iqm_cis = rly.get_interval_estimates(
        #     algo_scores, aggregate_func, reps=2000
        # )
        # sorted_algos = sorted(
        #     iqm_scores.keys(),
        #     key=lambda x: iqm_scores[x][0],
        #     reverse=True
        # )
        sorted_algos = sorted(
            algo_scores.keys(),
            key=lambda x: algo_scores[x][0],
            reverse=True
        )
        row_str = f"{env:<10} | "
        for rank, algo in enumerate(sorted_algos, 1):
            # score = iqm_scores[algo][0]
            # low = iqm_cis[algo][0][0]
            # high = iqm_cis[algo][1][0]
            score = algo_scores[algo][0]
            low = score - algo_scores[algo][1]
            high = score + algo_scores[algo][1]
            cell = f"[{rank}] {algo}: {score:.3f}({low:.3f}-{high:.3f})"
            row_str += f"\n{cell}"
        print(row_str.rstrip(" >> "))
        print("-" * 100)
if __name__ == "__main__":
    discrete = False
    Metric = "avg_reward"
    main()