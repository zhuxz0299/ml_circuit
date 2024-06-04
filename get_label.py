import os
import pickle
from tqdm import tqdm

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
            with open('train_label/' + state + '.pkl', 'wb') as f:
                pickle.dump(target, f)
