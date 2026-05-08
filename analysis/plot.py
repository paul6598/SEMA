import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve_tight(data_folder, save_folder, env_name, algos, legend_labels, metric="avg_reward"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(algos))

    for idx, algo in enumerate(algos):
        csv_path = os.path.join(data_folder, env_name, metric, f"{algo}.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: Data for {algo} in {env_name} not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        steps = df['Step'].values

        mean_cols = [col for col in df.columns if col != "Step" and "__MIN" not in col and "__MAX" not in col]
        min_cols = [col for col in df.columns if "__MIN" in col]
        max_cols = [col for col in df.columns if "__MAX" in col]

        mean_values = df[mean_cols].mean(axis=1)
        
        if min_cols and max_cols:
            lower_bound = df[min_cols].min(axis=1)
            upper_bound = df[max_cols].max(axis=1)
        else:
            lower_bound = df[mean_cols].min(axis=1)
            upper_bound = df[mean_cols].max(axis=1)

        # 메인 라인 (선 두께도 2.5로 살짝 키워 눈에 더 잘 띄게 조정)
        label = legend_labels.get(algo, algo.upper())
        ax.plot(steps, mean_values, label=label, color=colors[3-idx], linewidth=2.5)
        ax.fill_between(
            steps, lower_bound, upper_bound, color=colors[3-idx], alpha=0.2
        )

    ax.set_xlabel('Environment Steps(6x10^4)', fontsize=18)
    ax.set_ylabel('Episode Reward Mean', fontsize=18)
    ax.set_title(f'{env_name.replace("_", " ").upper()}', fontsize=22, pad=50, fontweight='bold')
    
    #제목 위치 
    # x축, y축 눈금 숫자 크기 확대
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xlim(0, 69)
    ax.set_ylim(2, 9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(algos),
        fontsize=16, # 범례 글씨 확대
        frameon=True,
        borderaxespad=0.
    )
    
    plt.tight_layout()
    
    # ✨ 2. PDF로 렌더링 및 저장 ✨
    # 확장자를 .pdf로 변경하고 format='pdf'를 명시했습니다.
    # PDF는 벡터 이미지라 확대해도 깨지지 않으므로 논문에 삽입하기에 완벽합니다.
    save_path = os.path.join(save_folder, f"{env_name}_{metric}_learning_curve.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')  # PNG도 저장 (논문 제출용)
    plt.show()

def main():
    data_folder = "./data" 
    save_folder = "./plots"
    os.makedirs(save_folder, exist_ok=True)
    envs = ["give_way"] 
    algos = ["mappo", "l2m2_wo_subreward", "l2m2", "sema"] 
    legend_labels = {
        "mappo": "MAPPO",
        "sema": "SEMA (Ours)",
        "l2m2": "L2M2",
        "l2m2_wo_subreward": "L2M2 w/o Sub-reward"
    }
    metric = "avg_reward"

    for env in envs:
        plot_learning_curve_tight(data_folder, save_folder, env, algos, legend_labels, metric)

if __name__ == "__main__":
    main()