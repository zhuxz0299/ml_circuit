import torch
import numpy as np
import abc_py
import os
from tqdm import tqdm

def get_graph(aig_file):
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(aig_file)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []

    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2

        if nodeType == 4:
            data['num_inverted_predecessors'][nodeIdx] = 1
        if nodeType == 5:
            data['num_inverted_predecessors'][nodeIdx] = 2

        if aigNode.hasFanin0():
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        
        if aigNode.hasFanin1():
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes
    return data
    

graph_folder = 'train_graph'
aig_folder = 'train_aig'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)
for aig_file in tqdm(os.listdir(aig_folder)):
    if aig_file.endswith('.aig'):
        data = get_graph(os.path.join(aig_folder, aig_file))
        graph_file = os.path.join(graph_folder, aig_file.replace('.aig', '.pt'))
        torch.save(data, graph_file)
