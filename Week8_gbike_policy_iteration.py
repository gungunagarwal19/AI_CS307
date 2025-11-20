#!/usr/bin/env python3

import numpy as np
from math import exp, factorial
from typing import Tuple, List

POISSON_CUTOFF = 11

def poisson_pmf(lmbda: float, n: int) -> float:
    return exp(-lmbda) * (lmbda ** n) / factorial(n)

def truncated_poisson_probs(lmbda: float, cutoff: int = POISSON_CUTOFF) -> List[float]:
    probs = [poisson_pmf(lmbda, n) for n in range(cutoff)]
    tail = 1.0 - sum(probs)
    probs[-1] += tail
    return probs

class GBike:
    def __init__(
        self,
        max_bikes: int = 20,
        rent_reward: int = 10,
        move_cost: int = 2,
        free_shuttle: int = 1,
        req_lambdas: Tuple[float, float] = (3.0, 4.0),
        ret_lambdas: Tuple[float, float] = (3.0, 2.0),
        gamma: float = 0.9,
        max_move: int = 5,
    ):
        self.max_bikes = max_bikes
        self.rent_reward = rent_reward
        self.move_cost = move_cost
        self.free_shuttle = free_shuttle
        self.req_lambdas = req_lambdas
        self.ret_lambdas = ret_lambdas
        self.gamma = gamma
        self.max_move = max_move

        self.req_probs = [truncated_poisson_probs(l) for l in req_lambdas]
        self.ret_probs = [truncated_poisson_probs(l) for l in ret_lambdas]

    def feasible_actions(self, i: int, j: int):
        actions = range(-self.max_move, self.max_move + 1)
        out = []
        for a in actions:
            if a > 0 and a > i:
                continue
            if a < 0 and -a > j:
                continue
            out.append(a)
        return out

    def expected_return(self, state, action, V):
        i, j = state

        if action > 0:
            move_cost = max(0, action - self.free_shuttle) * self.move_cost
        else:
            move_cost = abs(action) * self.move_cost

        immediate = -move_cost

        after_i = min(max(i - action, 0), self.max_bikes)
        after_j = min(max(j + action, 0), self.max_bikes)

        exp_ret = 0.0
        for r1, p_r1 in enumerate(self.req_probs[0]):
            for r2, p_r2 in enumerate(self.req_probs[1]):
                p_req = p_r1 * p_r2
                rented1 = min(after_i, r1)
                rented2 = min(after_j, r2)

                reward_rent = (rented1 + rented2) * self.rent_reward
                b1 = after_i - rented1
                b2 = after_j - rented2

                for ret1, p_ret1 in enumerate(self.ret_probs[0]):
                    for ret2, p_ret2 in enumerate(self.ret_probs[1]):
                        p = p_req * p_ret1 * p_ret2
                        next_i = min(b1 + ret1, self.max_bikes)
                        next_j = min(b2 + ret2, self.max_bikes)

                        park_cost = 4 * int(next_i > 10) + 4 * int(next_j > 10)

                        total = immediate + reward_rent - park_cost
                        exp_ret += p * (total + self.gamma * V[next_i, next_j])

        return exp_ret

    def policy_evaluation(self, policy, V, tol=1e-3, max_iter=1000):
        for _ in range(max_iter):
            delta = 0.0
            V_new = V.copy()
            for i in range(self.max_bikes + 1):
                for j in range(self.max_bikes + 1):
                    a = int(policy[i, j])
                    V_new[i, j] = self.expected_return((i, j), a, V)
                    delta = max(delta, abs(V_new[i, j] - V[i, j]))
            V[:] = V_new
            if delta < tol:
                break
        return V

    def policy_iteration(self, eval_tol=1e-3, max_iter=1000):
        V = np.zeros((self.max_bikes + 1, self.max_bikes + 1))
        policy = np.zeros_like(V, dtype=int)

        stable = False
        while not stable:
            V = self.policy_evaluation(policy, V, eval_tol, max_iter)
            stable = True

            for i in range(self.max_bikes + 1):
                for j in range(self.max_bikes + 1):
                    old = policy[i, j]
                    best = old
                    best_val = -1e12

                    for a in self.feasible_actions(i, j):
                        val = self.expected_return((i, j), a, V)
                        if val > best_val:
                            best_val = val
                            best = a

                    policy[i, j] = best
                    if best != old:
                        stable = False
        return policy, V


if __name__ == "__main__":
    gb = GBike()
    print("Running policy iteration...")
    policy, V = gb.policy_iteration()

    print("\nPolicy slice (0–5 bikes each location):")
    for i in range(6):
        print(" ".join(f"{int(policy[i,j]):3d}" for j in range(6)))

    print("\nValue slice:")
    for i in range(6):
        print(" ".join(f"{V[i,j]:7.1f}" for j in range(6)))

    np.savetxt("gbike_policy.csv", policy, fmt="%d", delimiter=",")
    np.savetxt("gbike_value.csv", V, fmt="%.4f", delimiter=",")

    print("\nSaved gbike_policy.csv and gbike_value.csv")
