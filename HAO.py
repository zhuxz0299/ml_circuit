# Hierarchical Action Optimizer (HAO)
import os
import random
import glob
from inference import GNN, Inference
random.seed(42)
libFile = '/home/zxz/course-project/ml_integrated_circuit_design/project/lib/7nm/7nm.lib'
AIG_PATH0 = "/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4/alu4.aig"
AIG_PATH = "/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4/alu4.aig"
AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0]
logFile = f'/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/{AIG_NAME}/{AIG_NAME}.log'

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

model_dict_path = '/home/zxz/course-project/ml_integrated_circuit_design/project/checkpoints/best_model.pth' # model 1
infer=Inference(model_dict_path)

def model(aig_path):
    return infer.inf(aig_path)

def unique_items(candidates):
    seen_scores = set()
    unique_candidates = []

    # 移除 'score' 重复的条目
    for candidate in candidates:
        if candidate[1] not in seen_scores:
            seen_scores.add(candidate[1])
            unique_candidates.append(candidate)

    return unique_candidates

def delete_files(directory):
    # 构建文件匹配模式，确保至少有一个非空字符跟在 'alu4' 后面
    pattern = os.path.join(directory, 'alu4_*.aig')
    
    # 使用 glob 模块找到所有匹配的文件
    files_to_delete = glob.glob(pattern)
    
    # 遍历文件列表并删除它们
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

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
        runCmd = f"/home/zxz/course-project/ml_integrated_circuit_design/yosys/yosys-abc -c \"read {AIG_PATH0};{actions}read_lib {libFile};write {new_aig};print_stats\" > {logFile}"
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
        unique_candidates=unique_items(new_candidates+candidates)
        candidates = sorted(unique_candidates, key=lambda x: x[1], reverse=True)[:k]
        if level == L - 1 or cur_level + level == L-1:
            # 最后一层，选择得分最高的一个，并找到其原始操作
            best_aig, best_score, best_action = max(candidates, key=lambda x: x[1])
            return best_aig, best_score, best_action  # 返回最初层的操作


action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']

# 指定要搜索的目录
directory = '/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4'

# 调用函数
delete_files(directory)
# print(model(AIG_PATH))
# print(model("/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4/alu4_6.aig"))
# print(model("/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4/alu4_64.aig"))
# print(model("/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4/alu4_643.aig"))
last_score = 0
for i in range(10):
    _, best_score, best_action = find_best_actions(AIG_PATH, action_list, k=3, L=2, cur_level=i)
    if best_action is not None:
        AIG_PATH = apply_action(AIG_PATH,best_action)
    print(f"Step {i}: Best AIG: {AIG_PATH}, Best Score: {best_score}, Best Action: {best_action}")
    if best_action is None or (best_score == last_score and i>3):
        break
    last_score = best_score




# 指定要搜索的目录
directory = '/home/zxz/course-project/ml_integrated_circuit_design/project/search/HAO/alu4'

# 调用函数
delete_files(directory)