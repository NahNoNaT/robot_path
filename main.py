# main.py
"""
Main runner:
- Builds gridworld (from config.yaml)
- Demonstrates planners for one trip: plan route from start to one goal and back
- Optionally visualizes with pygame_viz
- Runs simple RL agent (Value Iteration) demo to compute policy (for small grids)
"""
import yaml
from env.gridworld import GridWorld
from planners import astar_grid, bfs_grid, dijkstra_grid
from visualization.pygame_viz import animate_path
from mdp.mdp_model import SimpleMDPModel
from rl_agents.value_iteration import ValueIterationAgent
from utils import set_seed
import argparse
import os

def load_config(path="E:\CN_AI\FA2025\REL\config\config.yaml"):
    with open(path,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def plan_and_run_demo(cfg):
    set_seed(cfg.get("random_seed", None))
    gw = GridWorld(size=cfg.get("grid_size",10),
                   num_goal_cells=cfg.get("num_goal_cells",10),
                   items_per_goal=cfg.get("items_per_goal",5),
                   obstacle_prob=cfg.get("obstacle_prob", 0.12),
                   seed=cfg.get("random_seed", None))
    print("Start:", gw.start)
    print("Goals:", list(gw.goal_cells.items())[:6], " total items:", gw.goals_remaining())
    # choose first goal
    goals = [g for g in gw.goal_cells.keys()]
    if not goals:
        print("No goals available")
        return
    g = goals[0]
    print("Planning path to goal", g)
    path = astar_grid(gw.grid, gw.start, g)
    if path:
        print("Path length", len(path))
        if cfg.get("visualize", True):
            animate_path(gw, path, fps=cfg.get("render_fps",2))
    else:
        print("No path found by A*")
    # run Value Iteration demo (small only)
    if cfg.get("run_value_iteration", False):
        mdp = SimpleMDPModel(gw, carry_capacity=cfg.get("carry_capacity",3))
        vi = ValueIterationAgent(mdp)
        # initial state: pos=start, carried=0, goals tuple
        goals_state = tuple([gw.items_per_goal]*len(mdp.goal_positions))
        start_state = (gw.start, 0, goals_state)
        pi, V = vi.run(start_state)
        print("Value iteration done. Policy size:", len(pi))
    print("Demo finished.")

if __name__ == "__main__":
    cfg = load_config()
    plan_and_run_demo(cfg)
