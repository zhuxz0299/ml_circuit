import numpy as np
import torch
import abc_py
import os

folder = 'train_aig'
for file in os.listdir(folder)[:1]:
    state = os.path.join(folder, file)
    # 1. 初始化 abc 接口
    _abc = abc_py.AbcInterface()

    # 2. 启动 abc 接口
    _abc.start()

    # 3. 读取状态
    _abc.read(state)

    # 4. 初始化数据字典
    data = {}

    # 5. 获取节点数量
    numNodes = _abc.numNodes()

    # 6. 初始化数据字典中的 'node_type' 和 'num_inverted_predecessors' 数组
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)

    # 7. 初始化边的源节点和目标节点索引列表
    edge_src_index = []
    edge_target_index = []

    # 10. 遍历每个节点
    for nodeIdx in range(numNodes):
        # 11. 获取 AIG 节点
        aigNode = _abc.aigNode(nodeIdx)
        
        # 12. 获取节点类型
        nodeType = aigNode.nodeType()
        
        # 13. 初始化反转前驱节点数量
        data['num_inverted_predecessors'][nodeIdx] = 0
        
        # 14-23. 根据节点类型设置 'node_type' 和 'num_inverted_predecessors'
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

        # 24-27. 如果节点有 fanin0，添加边的索引
        if aigNode.hasFanin0():
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        
        # 28-31. 如果节点有 fanin1，添加边的索引
        if aigNode.hasFanin1():
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    # 32. 将边的索引列表转换为张量并存储在数据字典中
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)

    # 33. 将 'node_type' 数组转换为张量
    data['node_type'] = torch.tensor(data['node_type'])

    # 34. 将 'num_inverted_predecessors' 数组转换为张量
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])

    # 35. 将节点数量存储在数据字典中
    data['nodes'] = numNodes

    # for key, value in data.items():
    #     if isinstance(value, torch.Tensor):
    #         print(key, value.size())
    #     else:
    #         print(key, value)
    print(data)

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
    print(data)