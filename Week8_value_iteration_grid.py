#!/usr/bin/env python3

import numpy as np
from typing import Dict, Tuple

Coord = Tuple[int, int]

class GridWorld:
    def __init__(
        self,
        cols: int = 4,
        rows: int = 3,
        terminals: Dict[Coord, float] = None,
        walls: Tuple[Coord, ...] = None,
        step_reward: float = -0.04,
        gamma: float = 0.99,
        p_intended: float = 0.8,
    ):
        self.C = cols
        self.R = rows
        self.terminals = terminals or {(4, 3): 1.0, (4, 2): -1.0}
        self.walls = set(walls or {(2, 2)})
        self.step_reward = step_reward
        self.gamma = gamma
        self.actions = {"U": (0, 1), "R": (1, 0), "D": (0, -1), "L": (-1, 0)}
        self.p_intended = p_intended
        self.p_side = (1.0 - p_intended) / 2.0

    def in_bounds(self, x: int, y: int) -> bool:
        return 1 <= x <= self.C and 1 <= y <= self.R

    def is_wall(self, s: Coord) -> bool:
        return s in self.walls

    def is_terminal(self, s: Coord) -> bool:
        return s in self.terminals

    def move(self, s: Coord, action: str) -> Coord:
        dx, dy = self.actions[action]
        nx, ny = s[0] + dx, s[1] + dy
        if not self.in_bounds(nx, ny) or self.is_wall((nx, ny)):
            return s
        return (nx, ny)

    def successors(self, s: Coord, action: str):
        if self.is_terminal(s):
            return [(1.0, s, self.terminals[s], True)]

        outcomes = []
        order = ["U", "R", "D", "L"]

        outcomes.append((self.p_intended, self.move(s, action)))
        i = order.index(action)
        left = order[(i - 1) % 4]
        right = order[(i + 1) % 4]
        outcomes.append((self.p_side, self.move(s, left)))
        outcomes.append((self.p_side, self.move(s, right)))

        agg = {}
        for p, s2 in outcomes:
            agg[s2] = agg.get(s2, 0.0) + p

        res = []
        for s2, p in agg.items():
            r = self.terminals[s2] if s2 in self.terminals else self.step_reward
            res.append((p, s2, r, s2 in self.terminals))
        return res

    def value_iteration(self, tol: float = 1e-4, max_iter: int = 10000):
        V = {
            (x, y): 0.0
            for x in range(1, self.C + 1)
            for y in range(1, self.R + 1)
            if (x, y) not in self.walls
        }

        for _ in range(max_iter):
            delta = 0.0
            V_new = V.copy()
            for s in list(V.keys()):
                if self.is_terminal(s):
                    V_new[s] = self.terminals[s]
                    continue

                best = -1e12
                for a in self.actions:
                    total = 0.0
                    for p, s2, r, _ in self.successors(s, a):
                        total += p * (r + self.gamma * V[s2])
                    best = max(best, total)

                delta = max(delta, abs(V_new[s] - V[s]))
                V_new[s] = best

            V = V_new
            if delta < tol:
                break

        policy = {}
        for s in V.keys():
            if self.is_terminal(s):
                policy[s] = None
                continue

            best_a = None
            best_v = -1e12
            for a in self.actions:
                total = 0.0
                for p, s2, r, _ in self.successors(s, a):
                    total += p * (r + self.gamma * V[s2])
                if total > best_v:
                    best_v = total
                    best_a = a
            policy[s] = best_a

        return V, policy


def print_value_grid(V, gw):
    for y in range(gw.R, 0, -1):
        row = ""
        for x in range(1, gw.C + 1):
            s = (x, y)
            if s in gw.walls:
                row += "  ####   "
            elif s in gw.terminals:
                row += f"  {gw.terminals[s]:+0.2f}  "
            else:
                row += f"{V[s]:8.3f} "
        print(row)
    print()


def print_policy_grid(policy, gw):
    for y in range(gw.R, 0, -1):
        row = ""
        for x in range(1, gw.C + 1):
            s = (x, y)
            if s in gw.walls:
                row += "  #  "
            elif s in gw.terminals:
                row += "  T  "
            else:
                row += f"  {policy[s] or ' '}  "
        print(row)
    print()


if __name__ == "__main__":
    terminals = {(4, 3): 1.0, (4, 2): -1.0}
    walls = {(2, 2)}
    rewards = [-2.0, 0.1, 0.02, 1.0]

    for r in rewards:
        print("=" * 60)
        print(f"Per-step reward r(s) = {r}")
        gw = GridWorld(4, 3, terminals, walls, r, 0.99)
        V, policy = gw.value_iteration()
        print("Value function:")
        print_value_grid(V, gw)
        print("Greedy policy:")
        print_policy_grid(policy, gw)
