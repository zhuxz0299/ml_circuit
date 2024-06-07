import os
import random
def model(path):
    #返回0-1间的随机数，用于检验搜索算法逻辑是否正确
    return random.random()

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
libFile = '/root/ml_circuit/lib/7nm/7nm.lib'

k = 3  
AIG_PATH = "/root/ml_circuit/InitialAIG/test/alu4.aig" # path to the AIG file
AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0] # = "alu4"
logFile = f'/root/ml_circuit/search/kgreedy/{AIG_NAME}/{AIG_NAME}.log'
baseline = model(AIG_PATH) # NOTE

candidates = [{"file": AIG_PATH, "score": baseline, "action": ""}]

for step in range(10):
    new_candidates = []
    for candidate in candidates:
        for i, cur_action in enumerate(action_list):
            # eg alu4_0.aig->alu4_03.aig
            childFile = candidate['file'][:-4] + str(i) + ".aig" if step > 0 else AIG_NAME + "_" + str(i) + ".aig" # eg 
            # eg refactor;->refactor;rewrite -z;
            actions = candidate['action'] + cur_action + ";"

            runCmd = "/root/yosys/build/yosys-abc -c \"read " + AIG_PATH + ";" + \
                actions + "; read_lib " + libFile + "; write " + childFile + "; print_stats\" > " + logFile
            os.system(runCmd)

            score = model(childFile)  # NOTE 
            new_candidates.append({"file": childFile, "score": score, "action": actions})
    
    # 从 new_candidates 中选出前 k 个最优解
    candidates = sorted(new_candidates, key=lambda x: x['score'], reverse=True)[:k]
    print(f"step {step}: {candidates}")
# 选择最终的最优解
best_solution = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
print(f"Best solution: {best_solution}")

