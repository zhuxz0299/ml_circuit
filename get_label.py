import os
import pickle
from tqdm import tqdm

def get_label_task1():
    folder_path = 'project_data'
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    for filename in tqdm(pkl_files, desc="Processing .pkl files"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            inputs = data['input']
            targets = data['target']
            assert len(inputs) == len(targets), "Input and target lists must have the same length."
            
            for state, target in zip(inputs, targets):
                with open('tmp_data/train_label/' + state + '.pkl', 'wb') as f:
                    pickle.dump(target, f)

def get_label_task2():
    folder_path = 'project_data2'
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    for filename in tqdm(pkl_files, desc="Processing .pkl files"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            inputs = data['input']
            targets = data['target']
            assert len(inputs) == len(targets), "Input and target lists must have the same length."
            
            for state, target in zip(inputs, targets):
                with open('tmp_data/train_label_task2/' + state + '.pkl', 'wb') as f:
                    pickle.dump(target, f)

if __name__ == '__main__':
    get_label_task2()