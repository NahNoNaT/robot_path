# rl_agents/value_iteration.py
"""
Tabular Value Iteration over a small MDP built from mdp/mdp_model.py
Note: This is a simplistic implementation for demonstration.
"""
import math
from collections import defaultdict

class ValueIterationAgent:
    def __init__(self, mdp_model, gamma=0.99, theta=1e-3, max_iters=5000):
        self.mdp = mdp_model
        self.gamma = gamma
        self.theta = theta
        self.max_iters = max_iters
        self.V = {}  # dict for values keyed by state
        self.pi = {}

    def run(self, start_state, max_steps=10000):
        # We'll do asynchronous exploration-limited VI using simple back-up on visited states from BFS enum
        # For demonstration we do Monte Carlo style enumeration by BFS from start considering limited depth
        from collections import deque
        visited = set()
        q = deque([start_state])
        visited.add(start_state)
        while q:
            s = q.popleft()
            # expand neighbors via mdp actions
            for action in self.mdp.actions:
                ns, _ = self.mdp.step(s, action)
                if ns not in visited:
                    visited.add(ns)
                    q.append(ns)
            # also stop if too many states
            if len(visited) > 20000:
                break
        # initialize V
        for s in visited:
            self.V[s] = 0.0
        it = 0
        while True:
            delta = 0
            for s in list(visited):
                if self.mdp.is_terminal(s[2]):
                    continue
                best = -1e9
                for a in self.mdp.actions:
                    ns, r = self.mdp.step(s,a)
                    val = r + self.gamma * self.V.get(ns, 0.0)
                    if val > best:
                        best = val
                delta = max(delta, abs(self.V[s] - best))
                self.V[s] = best
            it += 1
            if delta < self.theta or it >= self.max_iters:
                break
        # extract greedy policy for visited states
        for s in visited:
            if self.mdp.is_terminal(s[2]):
                self.pi[s] = (0,0)
                continue
            best_a = None
            best_v = -1e9
            for a in self.mdp.actions:
                ns, r = self.mdp.step(s,a)
                val = r + self.gamma * self.V.get(ns, 0.0)
                if val > best_v:
                    best_v = val
                    best_a = a
            self.pi[s] = best_a
        return self.pi, self.V
