import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym

def make_env():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P        = env.unwrapped.P
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    def _cell_str(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)
    map_grid  = ["".join(_cell_str(cell) for cell in row)
                 for row in env.unwrapped.desc.tolist()]
    return P, n_states, n_actions, map_grid, env

def policy_evaluation(P, policy, n_states, gamma):
    P_pi = np.zeros((n_states, n_states))
    R_pi = np.zeros(n_states)

    for s in range(n_states):
        a = policy[s]  
        for (prob, next_s, reward, _) in P[s][a]:
            P_pi[s, next_s] += prob         
            R_pi[s]         += prob * reward  

    A = np.eye(n_states) - gamma * P_pi

    V = np.linalg.solve(A, R_pi)
    return V


def policy_improvement(P, V, n_states, n_actions, gamma):
    new_policy = np.zeros(n_states,)

    for s in range(n_states):
        q_sa = np.zeros(n_actions)
        for a in range(n_actions):
            for (prob, next_s, reward, terminated) in P[s][a]:
                bootstrap = 0.0 if terminated else gamma * V[next_s]
                q_sa[a] += prob * (reward + bootstrap)
        new_policy[s] = np.argmax(q_sa)  

    return new_policy


def policy_iteration(P, n_states, n_actions, gamma):
    policy = np.zeros(n_states, dtype=int)

    iters = 0

    while True:
        iters += 1
        V = policy_evaluation(P, policy, n_states, gamma)
        new_policy = policy_improvement(P, V, n_states, n_actions, gamma)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return V, policy, iters

ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def print_value_table(V, label="V*", nrow=4, ncol=4):
    print(f"\n── {label} table (4×4 grid) " + "─" * 36)
    print("       " + "  ".join(f"col{c}" for c in range(ncol)))
    for r in range(nrow):
        vals = "  ".join(f"{V[r * ncol + c]:8.5f}" for c in range(ncol))
        print(f"  row{r}  {vals}")


def print_policy(policy, map_grid, label="π*", nrow=4, ncol=4):
    print(f"\n── {label} policy (arrows) " + "─" * 37)
    for r in range(nrow):
        row_str = ""
        for c in range(ncol):
            cell = map_grid[r][c]
            row_str += f"  {'  ' if cell in ('H','G') else ACTION_ARROWS[policy[r*ncol+c]]} "
            if cell in ("H", "G"):
                row_str = row_str[:-3] + f"  {cell} "
        print(row_str)

if __name__ == "__main__":
    GAMMA = 0.99

    P, n_states, n_actions, map_grid, env = make_env()

    print("=" * 60)
    print("Policy Iteration with γ = 0.99")

    V_pi, pi_star, n_iters = policy_iteration(P, n_states, n_actions, GAMMA)

    print(f"\nConverged in {n_iters} iteration(s).")
    print_value_table(V_pi, label="V^π* (PI)")
    print_policy(pi_star, map_grid, label="π* (PI)")

    env.close()