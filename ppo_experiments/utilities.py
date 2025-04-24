import sys
import os
from os.path import dirname, abspath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from typing import Callable
from multiprocessing import Process
import torch
import gymnasium as gym
import ale_py
import csv
import json

gym.register_envs(ale_py)

currentFile = abspath(__file__)
currentFolder = dirname(currentFile)
parentFolder = dirname(currentFolder)
dir_containergym = parentFolder + '/containergym'
dir_optimizers = parentFolder + '/optimizers'

sys.path.append(dir_containergym)
sys.path.append(dir_optimizers)

from custom_actor_critic_policy import CustomActorCriticPolicy
from env import ContainerEnv
from adam_clara import Adam_CLARA
from sgd_clara import SGD_CLARA

# os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)


# TODO: Write proper docstrings for all methods
class LearningRateLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.gradients_path = os.path.join(log_dir, 'gradients.json')
        self.updates_path = os.path.join(log_dir, 'adam_updates.json')
        self.lr_path = os.path.join(log_dir, 'learning_rates.csv')
        
        self.gradients_buffer = []
        self.adam_update_buffer = []
        self.lr_buffer = []

        self.items_to_save_at_once = 5
    
    # Called right before training starts. Allows access to fully initialized model and environment.
    def _init_callback(self):
        # # Initialize gradients (steps added to current solution) JSON file if it does not exist
        # if not os.path.exists(self.gradients_path):
        #     with open(self.gradients_path, 'w') as f:
        #         json.dump([], f)  # Start with an empty list

        # Initialize Adam updates (steps added to current solution) JSON file if it does not exist
        self.adam_likes = [torch.optim.Adam, torch.optim.AdamW, torch.optim.Adamax, Adam_CLARA]
        if self.model.policy.optimizer.__class__ in self.adam_likes:
            if not os.path.exists(self.updates_path):
                with open(self.updates_path, 'w') as f:
                    json.dump([], f)  # Start with an empty list

        # Initialize learning rate csv with header
        with open(self.lr_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['timestep', 'learning_rate'])


    def get_gradients(self):
        stacked_gradients = []
        gradients = []
        for param in self.model.policy.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))  # Flatten each gradient
        if gradients:
            stacked_gradients = torch.cat(gradients).cpu().numpy()
        return stacked_gradients

    def get_adam_update(self):
        stacked_updates = []
        if self.model.policy.optimizer.__class__ in self.adam_likes:
            optimizer = self.model.policy.optimizer

            updates = []
            for param in self.model.policy.parameters():
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        m_t = state['exp_avg']
                        v_t = state['exp_avg_sq']
                        # step_size = optimizer.param_groups[0]['lr']  # TODO: Remove step_size
                        eps = optimizer.param_groups[0]['eps']

                        # Compute the Adam update step
                        update_step = -m_t / (torch.sqrt(v_t) + eps)  # TODO: Remove step_size
                        updates.append(update_step.view(-1))  # Flatten each update
                        
            if updates:
                stacked_updates = torch.cat(updates).cpu().numpy()

        return stacked_updates

    def get_lr_update(self):
        if 'd' in self.model.policy.optimizer.param_groups[0]:
            lr = self.model.policy.optimizer.param_groups[0]['lr'] * self.model.policy.optimizer.param_groups[0]['d']
        else:
            lr = self.model.policy.optimizer.param_groups[0]['lr']
        return lr
    
    def add_gradients_to_buffer(self, timestep, gradients):
        if len(gradients) > 0:
            json_entry = {
                        'timestep': timestep,
                        'gradients': gradients.tolist()
                    }
            self.gradients_buffer.append(json_entry)

    def add_adam_update_to_buffer(self, timestep, adam_update):
        if len(adam_update) > 0:
            json_entry = {
                        'timestep': timestep,
                        'adam_update': adam_update.tolist()
                    }
            self.adam_update_buffer.append(json_entry)

    def add_lr_update_to_buffer(self, timestep, lr_update):
        self.lr_buffer.append([timestep, lr_update])

    def write_gradients_buffer_to_disk(self):
        if self.gradients_buffer:
            # Read existing data
            with open(self.gradients_path, 'r') as f:
                data = json.load(f)
            # Append new array data
            data.extend(self.gradients_buffer)
            # Write back to JSON file
            with open(self.gradients_path, 'w') as f:
                json.dump(data, f)

    def write_adam_update_buffer_to_disk(self):
        if self.adam_update_buffer:
            # Read existing data
            with open(self.updates_path, 'r') as f:
                data = json.load(f)
            # Append new array data
            data.extend(self.adam_update_buffer)
            # Write back to JSON file
            with open(self.updates_path, 'w') as f:
                json.dump(data, f)

    def write_lr_update_buffer_to_disk(self):
        with open(self.lr_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(self.lr_buffer)
            
    def reset_gradients_buffer(self):
        self.gradients_buffer = []

    def reset_adam_update_buffer(self):
        self.adam_update_buffer = []

    def reset_lr_update_buffer(self):
        self.lr_buffer = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.model is None:
            return

        current_timestep = self.model.num_timesteps

        # # Get latest gradients and append to buffer
        # gradients = self.get_gradients()
        # self.add_gradients_to_buffer(current_timestep, gradients)

        # Get latest adam updates and append to buffer
        adam_updates = self.get_adam_update()
        self.add_adam_update_to_buffer(current_timestep, adam_updates)

        # Get latest learning rate and append to buffer
        lr_update = self.get_lr_update()
        self.add_lr_update_to_buffer(current_timestep, lr_update)

        # # If a buffer is full enough, write entire buffer to disk
        # if len(self.gradients_buffer) >= self.items_to_save_at_once:
        #     self.write_gradients_buffer_to_disk()
        #     self.reset_gradients_buffer()

        if len(self.adam_update_buffer) >= self.items_to_save_at_once:
            self.write_adam_update_buffer_to_disk()
            self.reset_adam_update_buffer()

        if len(self.lr_buffer) >= self.items_to_save_at_once:
            self.write_lr_update_buffer_to_disk()
            self.reset_lr_update_buffer()

    def _on_training_end(self):
        # # Write the remaining data in buffers to disk
        # self.write_gradients_buffer_to_disk()
        # self.reset_gradients_buffer()

        self.write_adam_update_buffer_to_disk()
        self.reset_adam_update_buffer()

        self.write_lr_update_buffer_to_disk()
        self.reset_lr_update_buffer()


class AdaptiveLearningRateCallback(BaseCallback):

    def __init__(self, lr0: float, d: float):
        super().__init__()
        self.lr = lr0
        self.path = 0
        self.c = 0.1  # Constant for cumul. path of Adam steps' variance
        self.d = d  # Damping factor used in learning rate update
        self.policy_update_count = 0
    
    def get_adam_update(self):
        stacked_updates = []
        if self.model.policy.optimizer.__class__ == torch.optim.Adam:
            optimizer = self.model.policy.optimizer

            updates = []
            for param in self.model.policy.parameters():
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        m_t = state['exp_avg']
                        v_t = state['exp_avg_sq']
                        eps = optimizer.param_groups[0]['eps']

                        # Compute the Adam update step
                        update_step = -m_t / (torch.sqrt(v_t) + eps)
                        updates.append(update_step.view(-1))  # Flatten each update
                        
            if updates:
                stacked_updates = torch.cat(updates).cpu().numpy()

        return stacked_updates

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.model is None:
            return

        self.policy_update_count += 1  # Count policy updates

        stacked_updates = self.get_adam_update()
        D = len(stacked_updates)

        if len(stacked_updates) > 0:  # TODO: Better way to check that gradient isn't empty?
            # Update path variable
            self.path = (1 - self.c) * self.path + np.sqrt(self.c * (2 - self.c)) * stacked_updates

            # Update learning rate starting from second "iteration"
            if self.policy_update_count > 1:
                self.lr = self.lr * np.exp(self.d * ((np.linalg.norm(self.path)**2 / D) - 1))

                # Update learning rate for all optimizer parameter groups
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = self.lr

                self.model.learning_rate = float(self.lr)  # TODO: Check values in logger files
                self.model._setup_lr_schedule()  # Needed for lr change to be effective


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f'Num timesteps: {self.num_timesteps}')
                    print(
                        f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}')

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f'Saving new best model to {self.save_path}.zip')
                    self.model.save(self.save_path)

        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve', window_size=25):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    :param window_size: (int) size of the sliding window used to smoothen the reward curve
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window_size)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(
        f'{title} Smoothed (moving average with window size {window_size})')
    plt.show(block=True)


def read_monitor_file(log_folder, window_size=25):
    """

    :param log_folder: (str) experiment location to read monitor file from
    :param window_size: (int) size of the sliding window used to smoothen the reward values
    :return:
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window_size)
    # Truncate x
    x = x[len(x) - len(y):]

    return x, y


def plot_multiple_results(log_folders, title='Learning curves', window_size=25, save_location=None):
    """
    plot the results

    :param log_folders: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward')
    # ax.set_title(f'{title} Smoothed (moving average with window size {window_size})')
    ax.set_title(f'{title}')

    for log_folder in log_folders:
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=window_size)
        # Truncate x
        x = x[len(x) - len(y):]
        ax.plot(x, y)

    ax.grid(True)
    ax.legend([f'{seed:02d}' for seed in range(len(log_folders))])
    if not save_location:
        save_location = f'{dirname(dirname(log_folders[0]))}/{title}.png'
    else:
        save_location = f'{save_location}/{title}.png'
    fig.savefig(save_location)


def plot_learning_curves(dir_experiment, config_names, n_models=15):
    for config_name in config_names:
        config_name = config_name.split('.')[0]
        log_dirs = []
        for seed in range(n_models):
            log_dir = f'{dir_experiment}/{config_name}/{seed:02d}/'
            log_dirs.append(log_dir)
        plot_multiple_results(log_dirs)


def get_model_by_path(model_path, env, policy_kwargs=None):
    return PPO.load(model_path, env, policy_kwargs=policy_kwargs)


def write_results_to_txt(test_results, experiment_path):
    n_models = len(test_results)

    test_results_sorted = sorted(test_results, key=lambda result: result[1]['avg_reward'], reverse=True)
    best_result = test_results_sorted[0]
    median_result = test_results_sorted[n_models // 2]

    all_models_results_txt = f"Mean and std dev of all {n_models} models:\n{test_results}\n"
    best_model_txt = f"Best model (seed={best_result[0]}): {best_result[1]['avg_reward']} \u00B1 {best_result[1]['std_reward']}\n"
    median_model_txt = f"Median model (seed={median_result[0]}): {median_result[1]['avg_reward']} \u00B1 {median_result[1]['std_reward']}\n"

    with open(f'{experiment_path}/test_results.txt', 'w') as f:
        f.write(all_models_results_txt)
        f.write(best_model_txt)
        f.write(median_model_txt)


def get_containergym_env_by_config_name(dir_configs, config_name, test=False):
    if test:
        config_name = config_name + '_test'
    config_file = config_name + '.json'

    config_path = f'{dir_configs}/{config_file}'

    return ContainerEnv.from_json(config_path)


def train_on_containergym(model_path,
                          dir_configs,
                          config_name,
                          timesteps,
                          seed=None,
                          learning_rate=0.0003,
                          damping=1,  # TODO: Change default damping value
                          policy_kwargs=None,
                          adaptive_lr=False):
    print(f'Training new model with config {config_name}')

    os.makedirs(model_path, exist_ok=True)

    env = get_containergym_env_by_config_name(dir_configs, config_name)
    env = Monitor(env, model_path)

    _, _ = env.reset(seed=seed)

    # If policy_kwargs contains a 'net_arch' key, use custom Actor-Critic architecture
    if 'net_arch' in policy_kwargs:
        model = PPO(CustomActorCriticPolicy, env, seed=seed, learning_rate=learning_rate, verbose=0,
                    policy_kwargs=policy_kwargs)
    else:
        model = PPO(MultiInputActorCriticPolicy, env, seed=seed, learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs)

    callback_list = [SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=model_path, verbose=0), 
                     LearningRateLoggerCallback(log_dir=model_path)]
    
    if adaptive_lr:
        callback_list.append(AdaptiveLearningRateCallback(lr0=learning_rate, d=damping))

    callbacks = CallbackList(callback_list)

    model.learn(total_timesteps=timesteps, callback=callbacks)

    return model


def rollout_on_containergym(model, env, seed=None, gamma=1., deterministic=True, live_plot=False):
    # Reset environment
    observation, info = env.reset(seed=seed)

    # Data collection
    df_volumes = pd.DataFrame(columns=env.enabled_containers)
    df_volumes.loc[len(df_volumes)] = env.state.volumes
    actions = []
    rewards = []
    n_free_presses = [
        len(observation['Time presses will be free']) - np.count_nonzero(observation['Time presses will be free'])]
    cumulative_reward = 0
    timestep = 0

    # Live plotting of volumes, reward, and #available PUs per timestep
    # TODO: Decide on whether to add an action plot
    if live_plot:
        plt.ion()
        fig = plt.figure(figsize=(15, 12))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        # Setting y-axis ticks to integers for press availability plots
        # ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Rollout
    while True:
        action = model.predict(observation, deterministic=deterministic)[0]  # prediction yields action and state
        observation, reward, terminated, truncated, info = env.step(action)

        df_volumes.loc[len(df_volumes)] = env.state.volumes
        actions.append(action)
        rewards.append(reward)
        n_free_presses.append(
            len(observation['Time presses will be free']) - np.count_nonzero(observation['Time presses will be free']))
        cumulative_reward += gamma ** timestep * reward
        timestep += 1

        if live_plot:
            ax1.clear()
            df_volumes.plot(ax=ax1,
                            color=[env.color_code[x] for x in df_volumes.columns],
                            xlim=(0, env.state.episode_length),
                            ylim=(0, max(env.max_volumes)),
                            xlabel='Timestep',
                            ylabel='Volume',
                            grid=True,
                            title='Containers volumes')

            ax2.clear()
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Reward')
            ax2.set_xlim(left=0, right=env.state.episode_length)
            ax2.set_ylim(bottom=env.r_pen, top=1.)
            ax2.grid(visible=True, which='major', axis='both')
            for j in range(env.state.episode_length):
                if actions[j] != 0:
                    ax2.scatter(j,
                                rewards[j],
                                color=env.color_code[env.enabled_containers[actions[j] - 1]],
                                clip_on=False)

            ax3.clear()
            ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax3.set_xlabel('Timestep')
            ax3.set_ylabel('Available PUs')
            ax3.set_xlim(left=0, right=env.state.episode_length)
            ax3.set_ylim(ymin=0, ymax=len(observation['Time presses will be free']))
            ax3.grid(visible=True, which='major', axis='both')
            ax3.plot(n_free_presses)

            plt.draw()
            plt.pause(0.0125)

        if terminated or truncated:
            break

        if live_plot:
            plt.show()

        env.close()

    if live_plot:
        # Annotate episodic cumulative reward
        ax2.annotate('Cumul. reward: {:.2f}'.format(sum(rewards)), xy=(0.9, 1.), xycoords='axes fraction')

    # Data that will be used for plotting
    return {'Cumulative reward': cumulative_reward,
            'Rollout length': timestep,
            'Volumes': df_volumes,
            'Actions': actions,
            'Rewards': rewards,
            'Available PUs': n_free_presses
            }


def evaluate_policy_on_containergym(model, env, n_rollouts, verbose=False):
    test_data = [rollout_on_containergym(model, env, seed=100 + i) for i in range(n_rollouts)]

    cumulative_rewards = np.array([rollout_data['Cumulative reward'] for rollout_data in test_data])
    rollout_lengths = np.array([rollout_data['Rollout length'] for rollout_data in test_data])

    avg_reward, std_reward = np.mean(cumulative_rewards), np.std(cumulative_rewards)
    avg_length, std_length = np.mean(rollout_lengths), np.std(rollout_lengths)

    if verbose:
        print(f'Average cumulative reward: {avg_reward} \u00B1 {std_reward}')
        print(f'Average episode length: {avg_length} \u00B1 {std_length}')

    return {'avg_reward': avg_reward, 'std_reward': std_reward, 'avg_length': avg_length, 'std_length': std_length}


def evaluate_experiment_on_containergym(experiment_path, dir_configs, config_name, n_models, policy_kwargs):
    print(f'Testing models with config {config_name}')
    env = get_containergym_env_by_config_name(dir_configs, config_name, test=True)

    test_results = []
    for seed in range(n_models):
        model_path = f'{experiment_path}/{seed:02d}/best_model'
        model = get_model_by_path(model_path, env, policy_kwargs)

        result = evaluate_policy_on_containergym(model, env, n_rollouts=15, verbose=True)

        test_results.append((seed, result))

    write_results_to_txt(test_results, experiment_path)

    return test_results


def experiment_on_containergym(dir_experiment,
                               dir_configs,
                               config_name,
                               timesteps,
                               n_models=15,
                               learning_rate=0.0003,
                               damping=1,
                               policy_kwargs=None,
                               adaptive_lr=False):
    experiment_path = f'{dir_experiment}/{config_name}'
    seeds = range(0, n_models)

    processes = [Process(target=train_on_containergym,
                         args=(f'{experiment_path}/{seed:02d}', dir_configs, config_name, timesteps, seed,
                               learning_rate, damping, policy_kwargs, adaptive_lr))
                 for seed in seeds]

    print(f'Starting experiment using config {config_name} and training for {int(timesteps)} timesteps!')

    # Parallelize experiments (one process per seed)
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    evaluate_experiment_on_containergym(experiment_path, dir_configs, config_name, n_models, policy_kwargs)


def run_all_experiments_on_containergym(dir_experiment,
                                        dir_configs,
                                        config_names,
                                        policy_kwargs,
                                        timesteps=None,
                                        n_models=15,
                                        learning_rate=0.0003,
                                        damping=1,
                                        default_actor_critic_arch=False,
                                        adaptive_lr=False):
    # Policy parameters to use in PPO
    net_arch = None
    budget = timesteps

    for config_name in config_names:
        if timesteps is None:
            if '11' in config_name:  # 11 container setup
                budget = 1e7  # 5e6
            elif '5' in config_name:  # 5 container setup
                budget = 5e6  # 2e6

        if '11' in config_name:
            net_arch = [11]
        elif '5' in config_name:
            net_arch = [5]

        # Preparing arguments to pass to the optimizer if custom Actor-Critic architecture is used
        if not default_actor_critic_arch:
            policy_kwargs['net_arch'] = net_arch

        experiment_on_containergym(dir_experiment,
                                   dir_configs,
                                   config_name,
                                   budget,
                                   n_models,
                                   learning_rate,
                                   damping,
                                   policy_kwargs,
                                   adaptive_lr)


def train_on_gymnasium(model_path, env_name, timesteps, seed=None, learning_rate=0.0003, damping=1,  # TODO: Change default damping value
                       policy_kwargs=None, adaptive_lr=False):
    print(f'Training new model with config {env_name}')

    os.makedirs(model_path, exist_ok=True)

    if env_name == 'Pong-v5':
        env = gym.make('ALE/Pong-v5')
    else:
        env = gym.make(env_name)
    env = Monitor(env, model_path)

    _, _ = env.reset(seed=seed)

    model = PPO('MlpPolicy', env, seed=seed, learning_rate=learning_rate, verbose=0, policy_kwargs=policy_kwargs)

    callback_list = [SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=model_path, verbose=0), 
                     LearningRateLoggerCallback(log_dir=model_path)]
    
    if adaptive_lr:
        callback_list.append(AdaptiveLearningRateCallback(lr0=learning_rate, d=damping))

    callbacks = CallbackList(callback_list)

    model.learn(total_timesteps=timesteps, callback=callbacks)

    return model


def rollout_on_gymnasium(model, env, seed=None, gamma=1., deterministic=True, live_plot=False):
    # Reset environment
    observation, info = env.reset(seed=seed)

    # Data collection
    cumulative_reward = 0
    timestep = 0

    # Rollout
    while True:
        action = model.predict(observation, deterministic=deterministic)[0]  # Prediction yields action and state
        observation, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += gamma ** timestep * reward
        timestep += 1

        if terminated or truncated:
            break

        env.close()

        # TODO: Add rendering

    return {'Cumulative reward': cumulative_reward,
            'Rollout length': timestep
            }


def evaluate_policy_on_gymnasium(model, env, n_rollouts, verbose=False):
    test_data = [rollout_on_gymnasium(model, env, seed=100 + i) for i in range(n_rollouts)]

    cumulative_rewards = np.array([rollout_data['Cumulative reward'] for rollout_data in test_data])
    rollout_lengths = np.array([rollout_data['Rollout length'] for rollout_data in test_data])

    avg_reward, std_reward = np.mean(cumulative_rewards), np.std(cumulative_rewards)
    avg_length, std_length = np.mean(rollout_lengths), np.std(rollout_lengths)

    if verbose:
        print(f'Average cumulative reward: {avg_reward} \u00B1 {std_reward}')
        print(f'Average episode length: {avg_length} \u00B1 {std_length}')

    return {'avg_reward': avg_reward, 'std_reward': std_reward, 'avg_length': avg_length, 'std_length': std_length}


def evaluate_experiment_on_gymnasium(experiment_path, env_name, n_models, policy_kwargs):
    print(f'Testing models on {env_name}')
    env = gym.make(env_name)

    test_results = []
    for seed in range(n_models):
        model_path = f'{experiment_path}/{seed:02d}/best_model'
        model = get_model_by_path(model_path, env, policy_kwargs)

        result = evaluate_policy_on_gymnasium(model, env, n_rollouts=15, verbose=True)

        test_results.append((seed, result))

    write_results_to_txt(test_results, experiment_path)

    return test_results


def experiment_on_gymnasium(dir_experiment,
                            env_name,
                            timesteps=5e6,
                            n_models=15,
                            learning_rate=0.0003,
                            damping=1,
                            policy_kwargs=None,
                            adaptive_lr=False):
    experiment_path = f'{dir_experiment}/{env_name}'
    seeds = range(0, n_models)

    processes = [Process(target=train_on_gymnasium,
                         args=(f'{experiment_path}/{seed:02d}', env_name, timesteps, seed, learning_rate,
                               damping, policy_kwargs, adaptive_lr))
                 for seed in seeds]

    print(f'Starting experiment on {env_name} and training for {int(timesteps)} timesteps!')

    # Parallelize experiments (one process per seed)
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    evaluate_experiment_on_gymnasium(experiment_path, env_name, n_models, policy_kwargs)


def run_all_experiments_on_gymnasium(dir_experiment,
                                     env_names,
                                     policy_kwargs,
                                     timesteps=None,
                                     n_models=15,
                                     learning_rate=0.0003,
                                     damping=1,
                                     adaptive_lr=False):
    budget = timesteps
    for env_name in env_names:
        if not timesteps:
            if env_name == 'Acrobot-v1' or env_name == 'Pendulum-v1':
                budget = 1e6
            if env_name == 'LunarLander-v3' or env_name == 'BipedalWalker-v3':
                budget = 5e6
            if env_name == 'Pong-v5':
                budget = 1e7
            if env_name == 'Ant-v5':
                budget = 2e7
            if env_name == 'Humanoid-v5':
                budget = 5e7

        experiment_on_gymnasium(dir_experiment, env_name, budget, n_models, learning_rate, damping, policy_kwargs,
                                adaptive_lr)
