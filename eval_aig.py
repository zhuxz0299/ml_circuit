import os
import re
libFile = './lib/7nm/7nm.lib'
logFile = 'alu2.log'
# 1. 构建 abcRunCmd 命令
abcRunCmd = "./yosys -abc -c \"read " + AIG + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile

# 2. 使用 os.system 执行命令
os.system(abcRunCmd)

# 3. 打开 logFile 并读取内容
with open(logFile) as f:
    # 4. 使用正则表达式提取区域信息
    areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])

# 5. 计算评估值
eval = float(areaInformation[-9]) * float(areaInformation[-4])

# ------------------------------------------------------------
# 1. 定义 RESYN2_CMD 命令
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"

# 2. 构建 abcRunCmd 命令
abcRunCmd = "./yosys -abc -c \"read " + circuitPath + "; " + RESYN2_CMD + " read_lib " + libFile + "; write " + nextState + "; write_bench -l " + nextBench + "; map; topo; stime\" > " + logFile

# 3. 使用 os.system 执行命令
os.system(abcRunCmd)

# 4. 打开 logFile 并读取内容
with open(self.logFile) as f:
    # 5. 使用正则表达式提取区域信息
    areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])

# 6. 计算 baseline 值
baseline = float(areaInformation[-9]) * float(areaInformation[-4])

# 7. 计算 eval 值
eval = 1 - eval / baseline
