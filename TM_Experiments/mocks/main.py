#---------------------------------------------------------*\
# Title: 
# Author: 
#---------------------------------------------------------*/

import os
import subprocess
import sys

# Absoluten Pfad zum algorithmic-efficiency-Verzeichnis hinzufügen
ae_path = os.path.abspath('./submodules/algorithmic-efficiency')
sys.path.insert(0, ae_path)

# Define tasks and frameworks to test
tasks = ['mnist', 'cifar10']
framework = 'torch'

# Define optimizers: name -> path to submission.py
optimizers = {
    'adam': './algorithmic-efficiency/reference_algorithms/paper_baselines/adamw/pytorch/submission.py',
    'clara_adam': './src/clara_adam.py'
}

# Output dir
experiment_base = './experiments'

# Number of trials for tuning
num_trials = 3

for task in tasks:
    for opt_name, submission_path in optimizers.items():
        exp_name = f'{task}_{opt_name}'

        print(f"\n🚀 Running: {exp_name}")

        command = [
            'python3', 'algorithmic-efficiency/submission_runner.py',
            f'--workload={task}',
            f'--framework={framework}',
            f'--submission_path={submission_path}',
            f'--experiment_dir={experiment_base}',
            f'--experiment_name={exp_name}'
        ]

        # Run benchmark
        subprocess.run(command, check=True)


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\