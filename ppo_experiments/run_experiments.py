import sys
import os
from os.path import dirname, abspath
import torch
import argparse
from utilities import run_all_experiments_on_containergym, run_all_experiments_on_gymnasium, linear_schedule
import dadaptation
import prodigyopt


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
    dir_experiment = '/local/aatamna/ppo_'

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
    else:
        dir_experiment += 'adam_'
        optimizer_class = torch.optim.Adam

    # Prepare policy (incl. optimizer) arguments to pass to PPO
    policy_kwargs = dict(
        optimizer_class=optimizer_class
        # optimizer_kwargs=dict(
        #     log_every=adam_adapt_log_every
        # )
    )

    if environment == 0:  # ContainerGym
        config_names = ['5containers_2presses_timestep_2min',
                        '5containers_5presses_timestep_2min',
                        '11containers_2presses_timestep_2min',
                        '11containers_11presses_timestep_2min'
                        ]

        dir_experiment += 'containergym'

        if optimizer == 0 or optimizer == 1:
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
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
            run_all_experiments_on_containergym(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
                                                dir_configs=dir_configs,
                                                config_names=config_names,
                                                n_models=n_models,
                                                learning_rate=learning_rate,
                                                default_actor_critic_arch=False,
                                                policy_kwargs=policy_kwargs,
                                                adaptive_lr=True)
    if environment == 1:  # Gymnasium environments
        env_names = ['LunarLander-v3',
                     'BipedalWalker-v3',
                     # 'Pong-v5',
                     # 'Ant-v5',
                     # 'Humanoid-v5'
                     ]

        dir_experiment += 'gymnasium'

        if optimizer == 0 or optimizer == 1:
            run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
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
            run_all_experiments_on_gymnasium(dir_experiment=dir_experiment + '_lr0_' + str(learning_rate),
                                             env_names=env_names,
                                             n_models=n_models,
                                             learning_rate=learning_rate,
                                             policy_kwargs=policy_kwargs,
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
                              '4: Adam with adaptive lr'
                              )
                        )
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4,
                        help='Initial learning rate (default: 3e-4)')

    args = parser.parse_args()

    # Call main() with parsed arguments
    main(args.environment, args.optimizer, args.learning_rate)
