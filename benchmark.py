# benchmark.py
"""
Run simple benchmarks on planners and RL agents, save/plot results.
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from env.gridworld import GridWorld
from planners import bfs_grid, dijkstra_grid, astar_grid
from mdp.mdp_model import SimpleMDPModel
from rl_agents.value_iteration import ValueIterationAgent
from rl_agents.q_learning import QLearningAgent
from utils import set_seed

def run_planner_test(grid_size=10, trials=20, seed=42):
    set_seed(seed)
    pl_names = ['BFS','Dijkstra','A*']
    times = {n:[] for n in pl_names}
    lengths = {n:[] for n in pl_names}
    for t in range(trials):
        gw = GridWorld(size=grid_size, num_goal_cells=5, obstacle_prob=0.14, seed=seed+t)
        start = gw.start
        # pick a random goal
        goals = [g for g in gw.goal_cells.keys()]
        if not goals:
            continue
        goal = goals[0]
        # BFS
        t0 = time.perf_counter(); path = bfs_grid(gw.grid, start, goal); t1 = time.perf_counter()
        times['BFS'].append(t1-t0); lengths['BFS'].append(len(path) if path else np.nan)
        # Dijkstra
        t0 = time.perf_counter(); path = dijkstra_grid(gw.grid, start, goal); t1 = time.perf_counter()
        times['Dijkstra'].append(t1-t0); lengths['Dijkstra'].append(len(path) if path else np.nan)
        # A*
        t0 = time.perf_counter(); path = astar_grid(gw.grid, start, goal); t1 = time.perf_counter()
        times['A*'].append(t1-t0); lengths['A*'].append(len(path) if path else np.nan)
    # compute mean & std
    for name in pl_names:
        print(f"{name}: time mean {np.nanmean(times[name]):.5f}s std {np.nanstd(times[name]):.5f}, len mean {np.nanmean(lengths[name]):.2f}")
    # plot
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].bar(times.keys(), [np.nanmean(times[k]) for k in times.keys()], yerr=[np.nanstd(times[k]) for k in times.keys()])
    axes[0].set_title('Average Time')
    axes[1].bar(lengths.keys(), [np.nanmean(lengths[k]) for k in lengths.keys()], yerr=[np.nanstd(lengths[k]) for k in lengths.keys()])
    axes[1].set_title('Average Path Length')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_planner_test()
