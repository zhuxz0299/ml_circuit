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
    folder_path = 'project_data'
    folder_path2 = 'project_data2'
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    for filename in tqdm(pkl_files, desc="Processing .pkl files"):
        file_path = os.path.join(folder_path, filename)
        file_path2 = os.path.join(folder_path2, filename)
        
        # 同时打开两个文件
        with open(file_path, 'rb') as file1, open(file_path2, 'rb') as file2:
            data1 = pickle.load(file1)
            data2 = pickle.load(file2)
            
            inputs1 = data1['input']
            targets1 = data1['target']
            targets2 = data2['target']
            
            # 处理第一个文件的数据
            for state1, target1, target2 in zip(inputs1, targets1, targets2):
                with open('tmp_data/train_label_task2/' + state1 + '.pkl', 'wb') as f:
                    pickle.dump(target2 - target1, f)
                    
if __name__ == '__main__':
    get_label_task2()