import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(vectors, progress_labels, title_suffix, filename_suffix):
    if len(vectors) < 2:
        print(f"[{title_suffix}] 데이터가 너무 적어 t-SNE를 수행할 수 없습니다.")
        return

    vectors = np.array(vectors)
    progress_labels = np.array(progress_labels)
    

    print(f"[{title_suffix}] t-SNE 계산 중... (총 데이터 수: {len(vectors)})")
    perplexity_value = min(30, len(vectors) - 1)
    tsne_model = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    vectors_2d = tsne_model.fit_transform(vectors)


    plt.figure(figsize=(12, 8))
    colors = ['dodgerblue', 'orange', 'mediumseagreen', 'tomato', 'purple', 'gold', 'magenta', 'cyan',]
    
    unique_progress = sorted(list(set(progress_labels)))
    for prog in unique_progress:
        indices = np.where(progress_labels == prog)[0]
        if len(indices) == 0:
            continue

        color = colors[prog % len(colors)]
        
        plt.scatter(
            vectors_2d[indices, 0], vectors_2d[indices, 1],
            s = 100,
            c=color, marker='o', alpha=0.7, edgecolors='w', linewidth=0.5,
            label=f'Progress 0.{prog}'
        )

    plt.title('t-SNE Result', fontsize=30)
    plt.xlabel('t-SNE 1', fontsize=25)
    plt.ylabel('t-SNE 2', fontsize=25)
    plt.legend(title='Progress Level', fontsize=15, title_fontsize=18, loc='best')
        
    plt.grid(True, linestyle='--', alpha=0.5)
    output_filename = f'plots/tsne_{filename_suffix}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"'{output_filename}' 저장 완료.\n")


def main():

    agent_split = True 
    all_vectors = []
    all_progress = []
    all_agents = []

    for progress in [0, 7]:
        file_path = f"data/give_way/mlp_result/tsne_data_0.{progress}_new.jsonl"
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    

                    all_vectors.append(data['vector'][0])
                    all_progress.append(progress)
                    all_agents.append(1)

                    all_vectors.append(data['vector'][1])
                    all_progress.append(progress)
                    all_agents.append(2)
                    
                    count += 1
            print(f"[Progress 0.{progress}] 데이터 {count}개 로드 완료.")
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")

    print("-" * 40)
    vectors_np = np.array(all_vectors)
    progress_np = np.array(all_progress)
    agents_np = np.array(all_agents)
    if agent_split:
        idx_agent1 = np.where(agents_np == 1)[0]
        if len(idx_agent1) > 0:
            plot_tsne(vectors_np[idx_agent1], progress_np[idx_agent1], "Agent 1 Only", "Agent1_split")
                      
        idx_agent2 = np.where(agents_np == 2)[0]
        if len(idx_agent2) > 0:
            plot_tsne(vectors_np[idx_agent2], progress_np[idx_agent2], "Agent 2 Only", "Agent2_split")
    else:
        plot_tsne(vectors_np, progress_np, "Agents Combined", "Agents_Combined")


if __name__ == "__main__":
    main()