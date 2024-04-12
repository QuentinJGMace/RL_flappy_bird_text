import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import random
from agents import *

def grid_search_params(env, agent_type, param_grid, num_episodes=5000, max_score=100, seed=42):
    results = {}
    keys, values = zip(*param_grid.items())
    for i, v in enumerate(product(*values)):
        params = dict(zip(keys, v))
        # Only keeps a subset of params for the labels
        varying_params = ['epsilon', 'epsilon_decay', 'alpha', 'lambd']
        label = ', '.join([f'{k}: {v}' for k, v in params.items() if k in varying_params])

        print(f'Set {i+1} : Running {label}')
        if agent_type == "MC":
            agent = MCControlAgent(env, params)
        elif agent_type == "SL":
            agent = SarsaLambdaAgent(env, params)
        random.seed(seed)
        _, score_history, reward_history = agent.train(num_episodes=num_episodes, max_score=max_score)
        results[str(params)] = (score_history, reward_history)
    return results

def plot_results(results):
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    for params, (score_history, reward_history) in results.items():
        # Converts params from a string to a dictionnary
        params = eval(params)
        # Only keeps a subset of params for the labels
        varying_params = ['epsilon', 'epsilon_decay', 'alpha', 'lambd']
        label = ', '.join([f'{k}: {v}' for k, v in params.items() if k in varying_params])
        axs[0].plot(pd.Series(score_history).rolling(500).mean(), label=label)
        axs[1].plot(pd.Series(reward_history).rolling(500).mean(), label=label)
    axs[0].set_title('Score History')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].tick_params(axis='x', labelleft=True)
    axs[0].legend(loc='upper left')
    axs[1].set_title('Reward History')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Reward')
    axs[1].tick_params(axis='x', labelleft=True)
    axs[1].legend(loc='upper left')
    plt.show()

def plot_value_function(agent):
    # Works only for the base environnement in the basic settings
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    x = np.arange(0, 14)
    y = np.arange(-12, 12)
    X, Y = np.meshgrid(x, y)
    value_fall = np.zeros((len(x), len(y)))
    value_up = np.zeros((len(x), len(y)))
    for i in x:
        for j in y:
            value_fall[i, j] = agent.q_values[(i, j)][0]
            value_up[i, j] = agent.q_values[(i, j)][1]

    # Plots the value functions, both with the same color scale
    value_up /= max(np.max(np.abs(value_up)), np.max(np.abs(value_fall)))
    value_fall /= max(np.max(np.abs(value_up)), np.max(np.abs(value_fall)))
    assert (not np.allclose(value_fall, value_up))
    axs[0].imshow(value_up, cmap='viridis')
    #labels the axis
    axs[0].set_ylabel('Distance to pipe')
    axs[0].set_xlabel('Vertical distance to pipe (+12)')
    axs[0].set_title('Value function for going up')
    axs[1].imshow(value_fall, cmap='viridis')
    #labels the axis
    axs[1].set_ylabel('Distance to pipe')
    axs[1].set_xlabel('Vertical distance to pipe (+12)')
    axs[1].set_title('Value function for going down')
    plt.show()


def plot_policy(agent):
    # Works only for the base environnement in the basic settings

    x = np.arange(0, 14)
    y = np.arange(-12, 12)
    X, Y = np.meshgrid(x, y)
    policy = agent.getPolicy()
    values = np.zeros((len(x), len(y)))
    for i in x:
        for j in y:
            if policy[(i, j)] == 0:
                values[i, j] = 0
            else:
                values[i, j] = 1

    # Plots the value function Z
    plt.imshow(values, cmap='viridis')
    #labels the axis
    plt.ylabel('Distance to pipe')
    plt.xlabel('Vertical distance to pipe (+12)')
    plt.title('Policy (yellow for up, purple for down)')
    plt.show()

def get_best_params(results):
    best_params = None
    best_score = 0
    for params, (score_history, reward_history) in results.items():
        if np.mean(score_history[-100:]) > best_score:
            best_score = np.mean(score_history[-100:])
            best_params = params
    return eval(best_params)