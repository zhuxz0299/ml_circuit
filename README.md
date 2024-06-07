## 文件结构
### 已有文件夹
* `checkpoints`: 存放训练好的模型
* `InitialAIG`: 存放初始的AIG文件
* `lib`: 存放一些工具函数

### 待下载文件夹
* `project_data`: 存放项目 task1 数据，包含 .pkl 文件，用于基于 `InitialAIG` 中的 AIG 文件生成训练数据
* `project_data2`: 存放项目 task2 数据，包含 .pkl 文件，用于基于 `InitialAIG` 中的 AIG 文件生成训练数据

### 待生成文件夹
* `tmp_data`: 用于存放预处理得到的数据
* `data`: 用于存放能直接用于训练的数据集
    

## 处理流程
### task1
1. 运行 `get_aig.py`，将 `InitialAIG` 中的 AIG 文件通过 `project_data` 中的规则，生成训练数据，以 `.aig` 文件的形式存放在 `tmp_data/train_aig` 中
2. 运行 `get_graph.py`，将 `tmp_data/train_aig` 中的训练数据转换为图数据，以 `.pt` 文件的形式存放在 `tmp_data/train_graph` 中
3. 运行 `get_label.py`，将 `project_data` 中的标签数据转换为训练数据，以 `.pkl` 文件的形式存放在 `tmp_data/train_label` 中
4. 运行 `merge_data.py`，将 `tmp_data/train_graph` 和 `tmp_data/train_label` 中的数据合并。最终得到 `train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`，存放在 `data` 中。
5. 运行 `train_graph.py`，训练模型，模型存放在 `checkpoints` 中

### task2
#### 推理得到 .aig 性能
* 方法一：运行 `inference.py`，将使用 `checkpoints` 中的模型，对给定的 AIG 文件进行推理，得到性能数据。
* 方法二：运行 `eval_aig.py`，能够利用 `yosys` 对给定的 AIG 文件进行综合，得到性能数据。