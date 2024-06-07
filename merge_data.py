import os
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
import pickle
import random


def convert_dict_to_data(data_dict, label):
    node_type = data_dict['node_type'].float().unsqueeze(1)  # Add feature dimension and convert to float
    num_inverted_predecessors = data_dict['num_inverted_predecessors'].float().unsqueeze(1)  # Add feature dimension and convert to float
    node_features = torch.cat([node_type, num_inverted_predecessors], dim=1)
    edge_index = data_dict['edge_index']
    data = Data(x=node_features, edge_index=edge_index, y=label)
    return data
    

def merge_and_split_datasets(train_graph_folder, train_label_folder, output_folder):
    pt_files = sorted([os.path.join(train_graph_folder, f) for f in os.listdir(train_graph_folder) if f.endswith('.pt')])
    pkl_files = sorted([os.path.join(train_label_folder, f) for f in os.listdir(train_label_folder) if f.endswith('.pkl')])

    train_dataset, val_dataset, test_dataset = [], [], []

    for pt_file, pkl_file in tqdm(zip(pt_files, pkl_files), desc="Merging files", total=len(pt_files)):
        data_dict = torch.load(pt_file)
        with open(pkl_file, 'rb') as f:
            label = pickle.load(f)

        data = convert_dict_to_data(data_dict, label)

        rand_val = random.random()
        if rand_val < 0.6:
            train_dataset.append(data)
        elif rand_val < 0.8:
            val_dataset.append(data)
        else:
            test_dataset.append(data)

    os.makedirs(output_folder, exist_ok=True)
    torch.save(train_dataset, os.path.join(output_folder, 'train_dataset_task2.pt'))
    torch.save(val_dataset, os.path.join(output_folder, 'val_dataset_task2.pt'))
    torch.save(test_dataset, os.path.join(output_folder, 'test_dataset_task2.pt'))



if __name__ == "__main__":
    train_graph_folder = 'tmp_data/train_graph'
    train_label_folder = 'tmp_data/train_label_task2'
    output_folder = 'data'
    merge_and_split_datasets(train_graph_folder, train_label_folder, output_folder)
