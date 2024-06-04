import pickle

# 定义 .pkl 文件的路径
pkl_file_path = 'project_data/arbiter_5.pkl'

# 打开 .pkl 文件并读取内容
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)


input = data['input']
target = data['target']

# 打印读取到的数据
print(len(input))
print(len(target))
