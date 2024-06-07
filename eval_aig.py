import os
import re

def eval_aig(aig_path):
    abc_path = '../yosys/yosys-abc'
    libFile = './lib/7nm/7nm.lib'
    logFile = os.path.join('tmp_data/eval_log', os.path.basename(aig_path).split('.')[0] + '.log')

    abcRunCmd = abc_path + " -c \"read " + aig_path + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)

    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-9]) * float(areaInformation[-4])

    # get baseline
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    aig_basename = os.path.basename(aig_path).split('_')[0] + '.aig'
    aig_base_path = os.path.join('./InitialAIG/train/', aig_basename)
    base_logFile = os.path.join('tmp_data/eval_log', os.path.basename(aig_base_path).split('.')[0] + '.log')

    abcRunCmd = abc_path + " -c \"read " + aig_base_path + "; " + RESYN2_CMD + " read_lib " + libFile + " ; map; topo ; stime \" > " + base_logFile
    os.system(abcRunCmd)

    with open(base_logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    baseline = float(areaInformation[-9]) * float(areaInformation[-4])

    eval = 1 - eval / baseline
    return eval

if __name__ == '__main__':
    aig_path = 'tmp_data/train_aig/adder_0000341036.aig'
    eval = eval_aig(aig_path)
    print(eval)
