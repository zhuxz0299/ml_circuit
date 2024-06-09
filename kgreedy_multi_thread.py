import os
import glob
import random
from multiprocessing import Pool
from inference import GNN, Inference
from eval_aig import eval_aig
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
model_dict_path = '/home/zxz/course-project/ml_integrated_circuit_design/project/checkpoints/best_model.pth' # model 1

infer=Inference(model_dict_path)

def model(aig_path):
    return infer.inf(aig_path)
    #return eval_aig(aig_path)

def run_simulation(params):
    step, candidate, action, libFile, AIG_PATH, logFile = params
    childFile = candidate['file'][:-4] + str(action) + ".aig" if step!=0 else candidate['file'][:-4] + "_" + str(action) + ".aig"
    actions = candidate['action'] + synthesisOpToPosDic[action] + ";"
    runCmd = f"/home/zxz/course-project/ml_integrated_circuit_design/yosys/yosys-abc -c \"read {AIG_PATH};{actions}read_lib {libFile};write {childFile};print_stats\" > {logFile}"
    os.system(runCmd) #eg 从alu.aig->alu_01243.aig 生成该文件
    score = model(childFile)
    return {"file": childFile, "score": score, "action": actions}

def unique_items(candidates):
    seen_scores = set()
    unique_candidates = []

    # 移除 'score' 重复的条目
    for candidate in candidates:
        if candidate['score'] not in seen_scores:
            seen_scores.add(candidate['score'])
            unique_candidates.append(candidate)

    return unique_candidates

def main():
    libFile = '/home/zxz/course-project/ml_integrated_circuit_design/project/lib/7nm/7nm.lib'
    AIG_PATH = "/home/zxz/course-project/ml_integrated_circuit_design/project/search/kgreedy/alu4/alu4.aig"
    AIG_NAME = AIG_PATH.split('/')[-1].split('.')[0]
    logFile = f'/home/zxz/course-project/ml_integrated_circuit_design/project/search/kgreedy/{AIG_NAME}/{AIG_NAME}.log'
    baseline = 0 # model(AIG_PATH)
    candidates = [{"file": AIG_PATH, "score": baseline, "action": ""}]

    num_processes = 12
    pool = Pool(processes=num_processes)

    for step in range(10):
        task_params = [(step, candidate, i, libFile, AIG_PATH, logFile) for candidate in candidates for i in range(7)]
        results = pool.map(run_simulation, task_params)
        unique_candidates = unique_items(results + candidates)
        candidates = sorted(unique_candidates, key=lambda x: x['score'], reverse=True)[:3]
        
        print(f"step {step}: {candidates}")
        print('\n')

    best_solution = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    
    print(f"Best solution: {best_solution}")
    pool.close()
    pool.join()

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

if __name__ == '__main__':
    main()
    delete_files('/home/zxz/course-project/ml_integrated_circuit_design/project/search/kgreedy/alu4')