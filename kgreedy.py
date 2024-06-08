# import os
# import random
# def model(path):
#     #返回0-1间的随机数，用于检验搜索算法逻辑是否正确
#     return random.random()

# synthesisOpToPosDic={
#     0: 'refactor',
#     1: 'refactor -z',
#     2: 'rewrite',
#     3: 'rewrite -z',
#     4: 'resub',
#     5: 'resub -z',
#     6: 'balance',
# }
# action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']
# libFile = '/root/ml_circuit/lib/7nm/7nm.lib'

# k = 3  
# AIG_PATH = "/root/ml_circuit/InitialAIG/test/alu4.aig" # path to the AIG file
# AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0] # = "alu4"
# logFile = f'/root/ml_circuit/search/kgreedy/{AIG_NAME}/{AIG_NAME}.log'
# baseline = model(AIG_PATH) # NOTE

# candidates = [{"file": AIG_PATH, "score": baseline, "action": ""}]

# for step in range(10):
#     new_candidates = []
#     for candidate in candidates:
#         for i, cur_action in enumerate(action_list):
#             # eg alu4_0.aig->alu4_03.aig
#             childFile = candidate['file'][:-4] + str(i) + ".aig" if step > 0 else AIG_NAME + "_" + str(i) + ".aig" # eg 
#             # eg refactor;->refactor;rewrite -z;
#             actions = candidate['action'] + cur_action + ";"

#             runCmd = "/root/yosys/build/yosys-abc -c \"read " + AIG_PATH + ";" + \
#                 actions + "; read_lib " + libFile + "; write " + childFile + "; print_stats\" > " + logFile
#             os.system(runCmd)

#             score = model(childFile)  # NOTE 
#             new_candidates.append({"file": childFile, "score": score, "action": actions})
    
#     # 从 new_candidates 中选出前 k 个最优解
#     candidates = sorted(new_candidates, key=lambda x: x['score'], reverse=True)[:k]
#     print(f"step {step}: {candidates}")
# # 选择最终的最优解
# best_solution = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
# print(f"Best solution: {best_solution}")

import os
import random
from multiprocessing import Pool
from inference import GNN, inference

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

def model(path):
    # 返回0-1间的随机数，用于检验搜索算法逻辑是否正确
    return random.random()

def run_simulation(params):
    step, candidate, action, libFile, AIG_PATH, logFile = params
    childFile = candidate['file'][:-4] + str(action) + ".aig" if step!=0 else candidate['file'][:-4] + "_" + str(action) + ".aig"
    actions = candidate['action'] + synthesisOpToPosDic[action] + ";"
    runCmd = f"/root/yosys/build/yosys-abc -c \"read {AIG_PATH};{actions}read_lib {libFile};write {childFile};print_stats\" > {logFile}"
    os.system(runCmd) #eg 从alu.aig->alu_01243.aig 生成该文件
    score = model(childFile)
    return {"file": childFile, "score": score, "action": actions}

def main():
    model = GNN(in_channels=2, hidden_channels=64, out_channels=32)
    libFile = '/root/ml_circuit/lib/7nm/7nm.lib'
    AIG_PATH = "/root/ml_circuit/search/kgreedy/alu4/alu4.aig"
    AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0]
    logFile = f'/root/ml_circuit/search/kgreedy/{AIG_NAME}/{AIG_NAME}.log'
    baseline = model(AIG_PATH)
    candidates = [{"file": AIG_PATH, "score": baseline, "action": ""}]

    num_processes = 12
    pool = Pool(processes=num_processes)

    for step in range(10):
        task_params = [(step, candidate, i, libFile, AIG_PATH, logFile) for candidate in candidates for i in range(7)]
        results = pool.map(run_simulation, task_params)
        candidates = sorted(candidates + results, key=lambda x: x['score'], reverse=True)[:3]
        print(f"step {step}: {candidates}")
        print('\n')

    best_solution = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    print(f"Best solution: {best_solution}")
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()