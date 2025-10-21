# rl_agents/policy_iteration.py
"""
Simple policy iteration on small explored state-set starting from start state
"""
from collections import deque

class PolicyIterationAgent:
    def __init__(self, mdp_model, gamma=0.99, max_iters=100):
        self.mdp = mdp_model
        self.gamma = gamma
        self.max_iters = max_iters
        self.V = {}
        self.pi = {}

    def run(self, start_state):
        # build reachable state set from start (bounded)
        visited = set()
        q = deque([start_state])
        visited.add(start_state)
        while q:
            s = q.popleft()
            for a in self.mdp.actions:
                ns, _ = self.mdp.step(s,a)
                if ns not in visited:
                    visited.add(ns)
                    q.append(ns)
            if len(visited) > 20000:
                break
        # init uniform random policy
        for s in visited:
            self.pi[s] = self.mdp.actions[0]
            self.V[s] = 0.0
        it = 0
        while it < self.max_iters:
            # policy evaluation (iterative)
            while True:
                delta = 0
                for s in visited:
                    if self.mdp.is_terminal(s[2]):
                        continue
                    a = self.pi[s]
                    ns, r = self.mdp.step(s,a)
                    v = r + self.gamma * self.V.get(ns, 0.0)
                    delta = max(delta, abs(self.V[s] - v))
                    self.V[s] = v
                if delta < 1e-3:
                    break
            # policy improvement
            policy_stable = True
            for s in visited:
                if self.mdp.is_terminal(s[2]):
                    continue
                old_a = self.pi[s]
                best_a = old_a
                best_v = -1e9
                for a in self.mdp.actions:
                    ns, r = self.mdp.step(s,a)
                    val = r + self.gamma * self.V.get(ns, 0.0)
                    if val > best_v:
                        best_v = val
                        best_a = a
                self.pi[s] = best_a
                if old_a != best_a:
                    policy_stable = False
            if policy_stable:
                break
            it += 1
        return self.pi, self.V
