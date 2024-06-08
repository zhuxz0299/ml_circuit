import heapq
from inference import GNN, Inference
action_list=['refactor', 'refactor -z', 'rewrite', 'rewrite -z', 'resub', 'resub -z', 'balance']
eval_cur_dict_path = '/root/ml_circuit/checkpoints/best_model.pth' # model 1
predict_dict_path = '/root/ml_circuit/checkpoints/best_model_task2.pth' # model 2

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

def model_eval_cur(aig):
    return eval_cur(aig)

def model_predict(aig):
    return predict(aig)

def apply_action(aig, idx:int):
    return aig[:-4] + str(idx) + ".aig" if '_' in aig.split('/')[-1] else aig[:-4] + "_" + str(idx) + ".aig"



def a_star_search(initial_aig):
    open_set = []
    initial_cost = model_eval_cur(initial_aig)
    initial_heuristic = model_predict(initial_aig)
    initial_node = Node(initial_aig, cost=initial_cost, heuristic=initial_heuristic)
    heapq.heappush(open_set, (initial_node.total_cost, initial_node))
    visited = set()

    while open_set:
        _, current_node = heapq.heappop(open_set) # = (current_node.total_cost, current_node)

        if current_node.heuristic == 0:  # heuristic为0，意味着达到最优状态
            path = []
            while current_node:
                path.append((current_node.aig, current_node.action))
                current_node = current_node.parent
            return path[::-1]

        visited.add(current_node.aig)

        for idx, action in enumerate(action_list):
            new_aig = apply_action(current_node.aig, idx)
            if new_aig in visited:
                continue
            new_cost = model_eval_cur(new_aig)
            new_heuristic = model_predict(new_aig)
            new_node = Node(new_aig, new_cost, new_heuristic, current_node, action)
            heapq.heappush(open_set, (new_node.total_cost, new_node))

    return None  # 如果没有找到路径

# 使用示例
initial_aig = "/root/ml_circuit/search/A*/alu4/alu4.aig"
path = a_star_search(initial_aig)
print("Optimal path:", path)