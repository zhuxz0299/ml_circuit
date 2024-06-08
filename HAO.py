# Hierarchical Action Optimizer (HAO)
import os
import random
random.seed(42)
libFile = '/root/ml_circuit/lib/7nm/7nm.lib'
AIG_PATH = "/root/ml_circuit/search/HAO/alu4/alu4.aig"
AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0]
logFile = f'/root/ml_circuit/search/HAO/{AIG_NAME}/{AIG_NAME}.log'

synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']

def model(file_path):
    # 返回0-1间的随机数，用于检验搜索算法逻辑是否正确
    return random.random()

def apply_action(aig, idx:int):
    return aig[:-4] + str(idx) + ".aig" if '_' in aig.split('/')[-1] else aig[:-4] + "_" + str(idx) + ".aig"

def simulate_actions(aig, action_list):
    # 模拟对AIG应用所有操作，返回操作和得分的列表
    results = []
    for i, action in enumerate(action_list):
        new_aig = apply_action(aig, i)
        actions_idx = new_aig.split('_')[-1].split('.')[0] #= "02314"
        #NOTE
        actions = ''
        for a in actions_idx:
            actions += (synthesisOpToPosDic[int(a)] + '; ')
        runCmd = f"/root/yosys/build/yosys-abc -c \"read {AIG_PATH};{actions}read_lib {libFile};write {new_aig};print_stats\" > {logFile}"
        os.system(runCmd)
        score = model(new_aig)
        results.append((new_aig, score, i))
    return results

def find_best_actions(initial_aig:str, action_list, k, L, cur_level):
    # 初始化候选列表
    candidates = [(initial_aig, model(initial_aig), None)]
    # 执行L层搜索
    for level in range(L):
        new_candidates = []
        for aig, _, _ in candidates:
            # 为每个候选模拟所有操作
            results = simulate_actions(aig, action_list)
            new_candidates.extend(results)
        # 选择得分最高的k个新候选
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:k]
        if level == L - 1 or cur_level + level == L-1:
            # 最后一层，选择得分最高的一个，并找到其原始操作
            best_aig, best_score, best_action = max(candidates, key=lambda x: x[1])
            return best_action  # 返回最初层的操作


action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']

for i in range(10):
    best_initial_action = find_best_actions(AIG_PATH, action_list, k=3, L=3, cur_level=i)
    AIG_PATH = apply_action(AIG_PATH,best_initial_action)
    print(f"Step {i}: {AIG_PATH}")

# import os
# import glob

# def delete_files(directory):
#     # 构建文件匹配模式，确保至少有一个非空字符跟在 'alu4' 后面
#     pattern = os.path.join(directory, 'alu4?*.aig')
    
#     # 使用 glob 模块找到所有匹配的文件
#     files_to_delete = glob.glob(pattern)
    
#     # 遍历文件列表并删除它们
#     for file_path in files_to_delete:
#         try:
#             os.remove(file_path)
#             print(f"Deleted {file_path}")
#         except Exception as e:
#             print(f"Failed to delete {file_path}: {e}")

# # 指定要搜索的目录
# directory = '/root/ml_circuit/InitialAIG/test'

# # 调用函数
# delete_files(directory)