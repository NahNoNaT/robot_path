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
from visualization.pygame_viz import animate_path
from mdp.mdp_model import SimpleMDPModel
from rl_agents.value_iteration import ValueIterationAgent
from utils import set_seed

def load_config(path="E:\CN_AI\FA2025\REL\config\config.yaml"):
    with open(path,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def run_rl_demo(cfg):
    # Set random seed
    set_seed(cfg.get("random_seed", None))
    
    # Create gridworld
    gw = GridWorld(size=cfg.get("grid_size",10),
                  num_goal_cells=cfg.get("num_goal_cells",5), # Reduced for RL
                  items_per_goal=cfg.get("items_per_goal",3),
                  obstacle_prob=cfg.get("obstacle_prob", 0.1),
                  seed=cfg.get("random_seed", None))
    
    print("Start:", gw.start)
    print("Goals:", list(gw.goal_cells.items())[:6], " total items:", gw.goals_remaining())

    # Create MDP model and run Value Iteration
    mdp = SimpleMDPModel(gw, carry_capacity=cfg.get("carry_capacity",3))
    vi = ValueIterationAgent(mdp, 
                           gamma=cfg.get("gamma", 0.99),
                           theta=cfg.get("theta", 1e-3),
                           max_iters=cfg.get("max_iters", 1000))

    # Initial state: (position, items carried, goal states)
    goals_state = tuple([gw.items_per_goal]*len(mdp.goal_positions))
    start_state = (gw.start, 0, goals_state)
    
    # Run value iteration
    print("Running Value Iteration...")
    pi, V = vi.run(start_state)
    print(f"Value iteration complete. Policy size: {len(pi)}")

    # Generate path using policy
    if cfg.get("visualize", True):
        current_state = start_state
        path = [current_state[0]]
        rewards = []  # Track rewards
        steps = 0
        max_steps = cfg.get("max_steps", 500)
        
        # Create a copy of gridworld for simulation
        sim_gw = gw.copy()
        
        while steps < max_steps:
            if mdp.is_terminal(current_state[2]) and current_state[0] == gw.start:
                print("All items collected and returned to start.")
                break

            action = pi.get(current_state)
            if action is None:
                break
                
            # Get next state and reward
            next_state, reward = mdp.step(current_state, action)
            rewards.append(reward)
            
            # Update gridworld state to match MDP state
            pos, carried, goals = next_state
            for idx, goal_pos in enumerate(mdp.goal_positions):
                if goal_pos in sim_gw.goal_cells:
                    sim_gw.goal_cells[goal_pos] = goals[idx]
                    if goals[idx] == 0:
                        sim_gw.grid[goal_pos] = 0
            
            current_state = next_state
            path.append(current_state[0])
            steps += 1
        
        print(f"Path length: {len(path)}")
        animate_path(sim_gw, path, 
                    fps=cfg.get("render_fps", 4),
                    step_delay=cfg.get("step_delay", 0.3),
                    rewards=rewards)  # Pass rewards to visualization

    print("Demo finished.")

if __name__ == "__main__":
    cfg = load_config()
    run_rl_demo(cfg)
