import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym

def make_env():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    n_states  = env.observation_space.n   
    n_actions = env.action_space.n        

    def _cell_str(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)
    map_grid = ["".join(_cell_str(cell) for cell in row) for row in env.unwrapped.desc.tolist()]

    return env, P, n_states, n_actions, map_grid


def value_iteration(P, n_states, n_actions, gamma, theta):

    V = np.zeros(n_states)

    threshold = theta * (1.0 - gamma) / gamma

    iters = 0

    while True:
        iters += 1

        V_new = np.zeros(n_states)

        for s in range(n_states):
            q_sa = np.zeros(n_actions)
            for a in range(n_actions):
                for (prob, next_s, reward, terminated) in P[s][a]:
                    bootstrap = 0.0 if terminated else gamma * V[next_s]
                    q_sa[a] += prob * (reward + bootstrap)
            V_new[s] = np.max(q_sa)

        delta = np.max(np.abs(V_new - V))
        V = V_new  

        if delta < threshold:
            break  

    # ── Extract greedy policy π* ─────
    policy = np.zeros(n_states)
    for s in range(n_states):
        q_sa = np.zeros(n_actions)
        for a in range(n_actions):
            for (prob, next_s, reward, terminated) in P[s][a]:
                bootstrap = 0.0 if terminated else gamma * V[next_s]
                q_sa[a] += prob * (reward + bootstrap)
        policy[s] = np.argmax(q_sa)

    return V, policy, iters

ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def print_value_table(V, nrow=4, ncol=4):
    print("\n── V* table (4x4 grid) " + "─" * 40)
    header = "       " + "  ".join(f"col{c}" for c in range(ncol))
    print(header)
    for r in range(nrow):
        vals = "  ".join(f"{V[r * ncol + c]:8.5f}" for c in range(ncol))
        print(f"  row{r}  {vals}")


def print_policy(policy, map_grid, nrow=4, ncol=4):
    print("\n── π* policy (arrows) " + "─" * 41)
    for r in range(nrow):
        row_str = ""
        for c in range(ncol):
            cell = map_grid[r][c]
            if cell in ("H", "G"):
                row_str += f"  {cell} "
            else:
                row_str += f"  {ACTION_ARROWS[policy[r * ncol + c]]} "
        print(row_str)
    print()
    print("  Legend:  ← LEFT(0)  ↓ DOWN(1)  → RIGHT(2)  ↑ UP(3)")


if __name__ == "__main__":

    GAMMA = 0.99    
    THETA = 1e-4   

    env, P, n_states, n_actions, map_grid = make_env()

    threshold = THETA * (1 - GAMMA) / GAMMA
    print(f"Convergence threshold  θ(1−γ)/γ = {threshold:.2e}")

    print("\nRunning value_iteration:")
    V_star, pi_star, n_iters = value_iteration(
        P, n_states, n_actions, gamma=GAMMA, theta=THETA
    )
    print(f"Converged in {n_iters} iterations.")

    print_value_table(V_star)
    print_policy(pi_star, map_grid)
    
    env.close()