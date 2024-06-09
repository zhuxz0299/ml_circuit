import os
import random
import glob
from eval_aig import eval_aig
from inference import GNN, Inference

synthesisOpToPosDic={
    0: 'refactor',
    1: 'refactor -z',
    2: 'rewrite',
    3: 'rewrite -z',
    4: 'resub',
    5: 'resub -z',
    6: 'balance',
}
action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']
libFile = '/home/zxz/course-project/ml_integrated_circuit_design/project/lib/7nm/7nm.lib'

k = 3  
AIG_PATH = "/home/zxz/course-project/ml_integrated_circuit_design/project/search/kgreedy/alu4/alu4.aig" # path to the AIG file
AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0] # = "alu4"
logFile = f'/home/zxz/course-project/ml_integrated_circuit_design/project/search/kgreedy/{AIG_NAME}/{AIG_NAME}.log'
baseline = 0 # NOTE

candidates = [{"file": AIG_PATH, "score": baseline, "action": ""}]



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

model_dict_path = '/home/zxz/course-project/ml_integrated_circuit_design/project/checkpoints/best_model.pth' # model 1
infer=Inference(model_dict_path)

def model(aig_path):
    #return infer.inf(aig_path)
    return eval_aig(aig_path,logFile)

def unique_items(candidates):
    seen_scores = set()
    unique_candidates = []

    # 移除 'score' 重复的条目
    for candidate in candidates:
        if candidate['score'] not in seen_scores:
            seen_scores.add(candidate['score'])
            unique_candidates.append(candidate)

    return unique_candidates



for step in range(10):
    new_candidates = []
    for candidate in candidates:
        for i, cur_action in enumerate(action_list):
            # eg alu4_0.aig->alu4_03.aig
            childFile = candidate['file'][:-4] + str(i) + ".aig" if step > 0 else AIG_NAME + "_" + str(i) + ".aig" # eg 
            # eg refactor;->refactor;rewrite -z;
            actions = candidate['action'] + cur_action + ";"

            runCmd = "/home/zxz/course-project/ml_integrated_circuit_design/yosys/yosys-abc -c \"read " + AIG_PATH + ";" + \
                actions + "; read_lib " + libFile + "; write " + childFile + "; print_stats\" > " + logFile
            os.system(runCmd)

            score = model(childFile)  # NOTE 
            new_candidates.append({"file": childFile, "score": score, "action": actions})
    
    # 从 new_candidates 中选出前 k 个最优解
    unique_candidates = unique_items(new_candidates + candidates)
    candidates = sorted(unique_candidates, key=lambda x: x['score'], reverse=True)[:k]
    print(f"step {step}: {candidates}")

# 选择最终的最优解
best_solution = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
print(f"Best solution: {best_solution}")
delete_files('/home/zxz/course-project/ml_integrated_circuit_design/project/search/kgreedy/alu4')