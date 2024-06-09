import heapq
import os
import glob
from inference import GNN, Inference
action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']
eval_cur_dict_path = '/home/zxz/course-project/ml_integrated_circuit_design/project/checkpoints/best_model.pth' # model 1
predict_dict_path = '/home/zxz/course-project/ml_integrated_circuit_design/project/checkpoints/best_model_task2.pth' # model 2
PATH0 = "/home/zxz/course-project/ml_integrated_circuit_design/project/search/A*/apex1/apex1.aig"
libFile = '/home/zxz/course-project/ml_integrated_circuit_design/project/lib/7nm/7nm.lib'
AIG_NAME = PATH0.split('/')[-1].split('.')[0]
logFile = f'/home/zxz/course-project/ml_integrated_circuit_design/project/search/A*/{AIG_NAME}/{AIG_NAME}.log'
eval_cur=Inference(eval_cur_dict_path)
predict=Inference(predict_dict_path)

class Node:
    def __init__(self, aig, cost, heuristic, parent=None, action=None):
        self.aig = aig
        self.cost = cost  # g value
        self.heuristic = heuristic  # h value
        self.total_cost = -(cost + heuristic)  # Store negative value to simulate max heap
        self.parent = parent
        self.action = action
    def __lt__(self, other):
        return self.total_cost < other.total_cost or (self.total_cost == other.total_cost and self.cost > other.cost)
def model_eval_cur(aig):
    return eval_cur.inf(aig)

def model_predict(aig):
    return predict.inf(aig)

def apply_action(aig, idx:int):
    return aig[:-4] + str(idx) + ".aig" if '_' in aig.split('/')[-1] else aig[:-4] + "_" + str(idx) + ".aig"

def delete_files(directory):
    pattern = os.path.join(directory, 'apex1_*.aig')
    
    # 使用 glob 模块找到所有匹配的文件
    files_to_delete = glob.glob(pattern)
    
    # 遍历文件列表并删除它们
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def a_star_search(initial_aig):
    open_set = []
    initial_cost = model_eval_cur(initial_aig)
    initial_heuristic = model_predict(initial_aig)
    initial_node = Node(initial_aig, cost=initial_cost, heuristic=initial_heuristic)
    heapq.heappush(open_set, (initial_node.total_cost, initial_node))
    visited = set()
    iteration = 0
    early_stopping = False
    while open_set:
        _, current_node = heapq.heappop(open_set) # = (current_node.total_cost, current_node)
        # 迭代一定轮次后退出：
        cur,expect = current_node.cost, current_node.heuristic
        if iteration > 9 or early_stopping or expect<3e-3:
            path = []
            while current_node:
                path.append((current_node.aig, current_node.action))
                current_node = current_node.parent
            return path[::-1], cur, expect

        visited.add(current_node.aig)

        for idx, action in enumerate(action_list):
            new_aig = apply_action(current_node.aig, idx)
            if new_aig in visited:
                continue
            actions = ''
            actions_idx = new_aig.split('_')[-1].split('.')[0] #= "02314"
            for a in actions_idx:
                actions += (action_list[int(a)] + '; ')
            runCmd = f"/home/zxz/course-project/ml_integrated_circuit_design/yosys/yosys-abc -c \"read {PATH0};{actions}read_lib {libFile};write {new_aig};print_stats\" > {logFile}"
            os.system(runCmd)
            new_cost = model_eval_cur(new_aig)
            new_heuristic = model_predict(new_aig)
            new_node = Node(new_aig, new_cost, new_heuristic, current_node, action)
            if abs(new_cost + new_heuristic + current_node.total_cost)<1e-3:
                early_stopping = True
            heapq.heappush(open_set, (new_node.total_cost, new_node))
        iteration += 1
    return None  # 如果没有找到路径


directory = '/home/zxz/course-project/ml_integrated_circuit_design/project/search/A*/apex1'
delete_files(directory)

initial_aig = PATH0
path, cur, expect = a_star_search(initial_aig)
print("Optimal path:", path)
print("Cost:", cur)
print("Heuristic:", expect)

delete_files(directory)