import sys
import os
from os.path import dirname, abspath
import torch
import argparse
import dadaptation
import prodigyopt

currentFile = abspath(__file__)
currentFolder = dirname(currentFile)
parentFolder = dirname(currentFolder)
dir_containergym = parentFolder + '/containergym'
dir_optimizers = parentFolder + '/optimizers'

sys.path.append(dir_containergym)
sys.path.append(dir_optimizers)

from utilities import run_all_experiments_on_containergym, run_all_experiments_on_gymnasium, linear_schedule
from adam_clara import Adam_CLARA
from sgd_clara import SGD_CLARA


def main(environment, optimizer, lr0):
    current_file = abspath(__file__)
    current_folder = dirname(current_file)
    parent_folder = dirname(current_folder)
    dir_containergym = parent_folder + '/containergym'
    dir_configs = dir_containergym + '/configs'

    sys.path.append(dir_containergym)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)

    n_models = 5
    learning_rate = lr0  # 3e-3, 3e-5
    lr_coef_dadaptation = [1, 0.5, 1e-1, 1e-2]
    lr_coef_prodigy = [1, 0.5, 1e-1, 1e-2]
    damping = 0.05  # Damping for cumulative learning rate adaptation
    dir_experiment = 'ppo_'

    optimizer_kwargs = {}
    # Set optimizer to be used by PPO
    if optimizer == 0:  # Default Adam without lr adaptation
        dir_experiment += 'adam_'
        optimizer_class = torch.optim.Adam
    elif optimizer == 1:  # Default Adam with linear lr schedule
        dir_experiment += 'adam_linear_schedule_'
        optimizer_class = torch.optim.Adam
        learning_rate = linear_schedule(learning_rate)
    elif optimizer == 2:  # D-Adaptation lr adaptation
        dir_experiment += 'dadaptation_'
        optimizer_class = dadaptation.DAdaptAdam
    elif optimizer == 3:  # Prodigy lr adaptation
        dir_experiment += 'prodigy_'
        optimizer_class = prodigyopt.Prodigy
    elif optimizer == 4:
        dir_experiment += 'adam_adaptive_lr_'
        optimizer_class = torch.optim.Adam
    elif optimizer == 5:  # Adam CLARA
        dir_experiment += 'adam_clara_'
        optimizer_class = Adam_CLARA
    elif optimizer == 6:  # SGD CLARA
        dir_experiment += 'sgd_clara_'
        optimizer_class = SGD_CLARA
    else:
        dir_experiment += 'adam_'
        optimizer_class = torch.optim.Adam

    # Prepare policy (incl. optimizer) arguments to pass to PPO
    policy_kwargs = dict(
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs
    )

    if environment == 0:  # ContainerGym
        config_names = ['5containers_2presses_timestep_2min',
                        '5containers_5presses_timestep_2min',
                        '11containers_2presses_timestep_2min',
                        '11containers_11presses_timestep_2min'
                        ]

        dir_experiment += 'containergym'

        if optimizer == 0:
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
                                                dir_configs=dir_configs,
                                                config_names=config_names,
                                                n_models=n_models,
                                                learning_rate=learning_rate,
                                                default_actor_critic_arch=False,
                                                policy_kwargs=policy_kwargs)

        if optimizer == 1:
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + "_lr0_" + str(lr0),
                                                dir_configs=dir_configs,
                                                config_names=config_names,
                                                n_models=n_models,
                                                learning_rate=learning_rate,
                                                default_actor_critic_arch=False,
                                                policy_kwargs=policy_kwargs)

        if optimizer == 2:  # Set lr coefficient for D-Adaptation to force smaller (or larger) lr estimates
            for c in lr_coef_dadaptation:
                run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr_coef_' + str(c),
                                                    dir_configs=dir_configs,
                                                    config_names=config_names,
                                                    n_models=n_models,
                                                    learning_rate=c,
                                                    default_actor_critic_arch=False,
                                                    policy_kwargs=policy_kwargs)

        if optimizer == 3:  # Set lr coefficient for Prodigy to force smaller (or larger) lr estimates
            for c in lr_coef_prodigy:
                policy_kwargs['optimizer_kwargs'] = dict(d_coef=c)
                run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr_coef_' + str(c),
                                                    dir_configs=dir_configs,
                                                    config_names=config_names,
                                                    n_models=n_models,
                                                    learning_rate=1.,  # lr value recommended by Mishchenko et al.
                                                    default_actor_critic_arch=False,
                                                    policy_kwargs=policy_kwargs)
        if optimizer == 4:
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate) + '_d_' + str(damping),
                                                dir_configs=dir_configs,
                                                config_names=config_names,
                                                n_models=n_models,
                                                learning_rate=learning_rate,
                                                default_actor_critic_arch=False,
                                                policy_kwargs=policy_kwargs,
                                                damping=damping,
                                                adaptive_lr=True)
        if optimizer == 5:
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
                                                dir_configs=dir_configs,
                                                config_names=config_names,
                                                n_models=n_models,
                                                learning_rate=learning_rate,
                                                default_actor_critic_arch=False,
                                                policy_kwargs=policy_kwargs,
                                                adaptive_lr=False)
        if optimizer == 6:
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
                                                dir_configs=dir_configs,
                                                config_names=config_names,
                                                n_models=n_models,
                                                learning_rate=learning_rate,
                                                default_actor_critic_arch=False,
                                                policy_kwargs=policy_kwargs,
                                                adaptive_lr=False)
                
        
    if environment == 1:  # Gymnasium environments
        env_names = ['LunarLander-v3',
                     'BipedalWalker-v3',
                     'Acrobot-v1',
                     'Pendulum-v1',
                     # 'Pong-v5',
                     # 'Ant-v5',
                     # 'Humanoid-v5'
                     ]

        dir_experiment += 'gymnasium'

        if optimizer == 0:
            run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
                                             env_names=env_names,
                                             n_models=n_models,
                                             learning_rate=learning_rate,
                                             policy_kwargs=policy_kwargs)
        if optimizer == 1:
            run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr0_' + str(lr0),
                                             env_names=env_names,
                                             n_models=n_models,
                                             learning_rate=learning_rate,
                                             policy_kwargs=policy_kwargs)

        if optimizer == 2:  # Set lr coefficient for D-Adaptation to force smaller (or larger) lr estimates
            for c in lr_coef_dadaptation:
                run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr_coef_' + str(c),
                                                 env_names=env_names,
                                                 n_models=n_models,
                                                 learning_rate=c,
                                                 policy_kwargs=policy_kwargs)

        if optimizer == 3:  # Set lr coefficient for Prodigy to force smaller (or larger) lr estimates
            for c in lr_coef_prodigy:
                policy_kwargs['optimizer_kwargs'] = dict(d_coef=c)
                run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr_coef_' + str(c),
                                                 env_names=env_names,
                                                 n_models=n_models,
                                                 learning_rate=1.,  # lr value recommended by Mishchenko et al.
                                                 policy_kwargs=policy_kwargs)

        if optimizer == 4:
            run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate) + '_d_' + str(damping),
                                             env_names=env_names,
                                             n_models=n_models,
                                             learning_rate=learning_rate,
                                             policy_kwargs=policy_kwargs,
                                             damping=damping,
                                             adaptive_lr=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add two numbers from the command line.')
    parser.add_argument('-e', '--environment', type=int,
                        help='Environment type. 0: ContainerGym, 1: Gymnasium environments.')
    parser.add_argument('-o', '--optimizer', type=int,
                        help=('Optimizer to use in PPO. '
                              '0: Default Adam, '
                              '1: Adam with linear lr schedule, '
                              '2: D-Adaptation, '
                              '3: Prodigy, '
                              '4: Adam with adaptive lr, '
                              '5: Adam with CLARA, '
                              '6: SGD with CLARA'
                              )
                        )
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4,
                        help='Initial learning rate (default: 3e-4)')

    args = parser.parse_args()

    # Call main() with parsed arguments
    main(args.environment, args.optimizer, args.learning_rate)
