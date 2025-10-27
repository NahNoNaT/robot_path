# rl_agents/sarsa.py
import random
from collections import defaultdict

class SarsaAgent:
    def __init__(self, mdp_model, alpha=0.5, gamma=0.99, epsilon=0.1, episodes=2000, max_steps=500):
        self.mdp = mdp_model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps = max_steps
        self.Q = defaultdict(lambda: {a:0.0 for a in self.mdp.actions})

    def choose_action(self, state):
        import random
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            qvals = self.Q[state]
            return max(qvals.items(), key=lambda kv: kv[1])[0]

    def run(self, start_state):
        for ep in range(self.episodes):
            state = start_state
            a = self.choose_action(state)
            for t in range(self.max_steps):
                ns, r = self.mdp.step(state, a)
                a2 = self.choose_action(ns)
                self.Q[state][a] += self.alpha * (r + self.gamma * self.Q[ns][a2] - self.Q[state][a])
                state, a = ns, a2
                if self.mdp.is_terminal(state):
                    break
        pi = {}
        for s, actions in self.Q.items():
            best = max(actions.items(), key=lambda kv: kv[1])[0]
            pi[s] = best
        return pi, self.Q
