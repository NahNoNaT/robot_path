# rl_agents/q_learning.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, mdp_model, alpha=0.5, gamma=0.99, epsilon=0.1, episodes=2000, max_steps=500):
        self.mdp = mdp_model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps = max_steps
        self.Q = defaultdict(lambda: {a:0.0 for a in self.mdp.actions})

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            qvals = self.Q[state]
            return max(qvals.items(), key=lambda kv: kv[1])[0]

    def run(self, start_state, log_every=100, verbose=True):
        episode_rewards = []
        episode_lengths = []
        successes = 0
        for ep in range(self.episodes):
            state = start_state
            total = 0.0
            steps = 0
            for t in range(self.max_steps):
                a = self.choose_action(state)
                ns, r = self.mdp.step(state, a)
                best_next = max(self.Q[ns].values()) if ns in self.Q else 0.0
                self.Q[state][a] += self.alpha * (r + self.gamma * best_next - self.Q[state][a])
                total += r
                state = ns
                steps += 1
                if self.mdp.is_terminal(state):
                    successes += 1
                    break
            episode_rewards.append(total)
            episode_lengths.append(steps)
            if verbose and log_every and (ep + 1) % log_every == 0:
                window = episode_rewards[-log_every:]
                avg_recent = sum(window) / len(window)
                print(f"Episode {ep+1}/{self.episodes}, avg reward (last {len(window)}): {avg_recent:.2f}")
        # derive policy
        pi = {}
        for s, actions in self.Q.items():
            best = max(actions.items(), key=lambda kv: kv[1])[0]
            pi[s] = best
        metrics = {
            "episodes": self.episodes,
            "successes": successes,
            "success_rate": successes / self.episodes if self.episodes else 0.0,
            "avg_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
            "avg_reward_last_50": (sum(episode_rewards[-50:]) / min(len(episode_rewards), 50)) if episode_rewards else 0.0,
            "best_reward": max(episode_rewards) if episode_rewards else 0.0,
            "avg_steps": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
        }
        return pi, self.Q, metrics, episode_rewards, episode_lengths
