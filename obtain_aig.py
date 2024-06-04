import os
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_aig(state, target):
    # state = 'alu2_0130622'
    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = os.path.join('train_log', circuitName + '.log') 
    nextState = os.path.join('train_aig', state + '.aig') 

    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + '; ')

    abcRunCmd = (
        "../yosys/yosys-abc -c \"read " + circuitPath + "; " + action_cmd +
        "read_lib " + libFile + "; write " + nextState + "; print_stats \" > " + logFile
    )
    os.system(abcRunCmd)


folder_path = 'project_data'
pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
num_threads = os.cpu_count() * 2  # 可以根据需要调整线程数量

# 使用 ThreadPoolExecutor 创建线程池
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for filename in tqdm(pkl_files, desc="Processing .pkl files"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            # 确保 'input' 和 'target' 列表的长度相同
            inputs = data['input']
            targets = data['target']
            assert len(inputs) == len(targets), "Input and target lists must have the same length."
            
            for state, target in zip(inputs, targets):
                # 提交任务到线程池
                futures.append(executor.submit(generate_aig, state, target))

    # 等待所有任务完成
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating AIG files"):
        pass

print("All AIG files generated.")