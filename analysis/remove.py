# csv에서 특정 문자열이 들어간 열을 제거하는 스크립트
import os
import pandas as pd

def remove_columns_with_value(data_folder, env_name, metric, value_to_remove):
    csv_path = os.path.join(data_folder, env_name, metric, f"mappo.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: Data for mappo in {env_name} not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # value_to_remove이 포함된 열 이름을 찾습니다
    columns_to_remove = [col for col in df.columns if str(value_to_remove) in col]
    
    # 해당 열들을 제거합니다
    df.drop(columns=columns_to_remove, inplace=True)
    
    # 변경된 DataFrame을 다시 저장합니다
    df.to_csv(csv_path, index=False)
    print(f"Removed columns containing '{value_to_remove}' from {csv_path}")

if __name__ == "__main__":
    data_folder = "./data"
    env_name = "give_way"
    metric = "avg_reward"
    values_to_remove = ["seed4", "seed0"]  # 제거할 값 (예: 0.1이 포함된 열 제거)
    
    for value in values_to_remove:
        remove_columns_with_value(data_folder, env_name, metric, value)