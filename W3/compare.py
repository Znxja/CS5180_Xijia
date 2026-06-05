import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pi import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    print_value_table,
    print_policy,
    ACTION_ARROWS,
)

def make_env():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P         = env.unwrapped.P
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    def _cell_str(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)
    map_grid  = ["".join(_cell_str(cell) for cell in row)
                 for row in env.unwrapped.desc.tolist()]
    return P, n_states, n_actions, map_grid, env

def value_iteration_instrumented(P, n_states, n_actions, gamma, theta,
                                  store_iterates=False):
    threshold = theta * (1.0 - gamma) / gamma
    V       = np.zeros(n_states)
    history = []
    iters   = 0

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

        if store_iterates:
            pi_k = policy_improvement(P, V, n_states, n_actions, gamma)
            history.append((V.copy(), pi_k.copy()))

        if delta < threshold:
            break

    policy = policy_improvement(P, V, n_states, n_actions, gamma)

    return V, policy, iters, history

def run_comparison(P, n_states, n_actions, gammas, theta=1e-4):
    results = {}

    for gamma in gammas:
        print(f"\n── γ = {gamma} " + "─" * 50)
        row = {}

        t0 = time.perf_counter()
        V_vi, pi_vi, iters_vi, _ = value_iteration_instrumented(
            P, n_states, n_actions, gamma, theta, store_iterates=False)
        t_vi = time.perf_counter() - t0
        backups_vi = iters_vi * n_states**2 * n_actions

        row["vi"] = {"iters": iters_vi, "time": t_vi, "backups": backups_vi}
        print(f"  VI  iters={iters_vi:5d}   time={t_vi*1000:8.2f} ms   "
              f"backups={backups_vi:>12,}")

        t0 = time.perf_counter()
        V_pi, pi_pi, iters_pi = policy_iteration(P, n_states, n_actions, gamma)
        t_pi = time.perf_counter() - t0
        backups_pi = iters_pi * n_states**3   

        row["pi"] = {"iters": iters_pi, "time": t_pi, "backups": backups_pi}
        print(f"  PI  iters={iters_pi:5d}   time={t_pi*1000:8.2f} ms   "
              f"backups={backups_pi:>12,}")

        agree = np.array_equal(pi_vi, pi_pi)
        print(f"  Policies match: {agree}")

        results[gamma] = row

    return results


def plot_comparison(results, gammas, save_path="compare_iterations.png"):

    vi_iters = [results[g]["vi"]["iters"] for g in gammas]
    pi_iters = [results[g]["pi"]["iters"] for g in gammas]
    gamma_labels = [str(g) for g in gammas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(gammas))
    width = 0.35

    bars_vi = ax.bar(x - width/2, vi_iters, width, label="Value Iteration",
                     color="#3498db", alpha=0.85)
    bars_pi = ax.bar(x + width/2, pi_iters, width, label="Policy Iteration",
                     color="#e74c3c", alpha=0.85)

    for bar in bars_vi:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=8, color="#2c3e50")
    for bar in bars_pi:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=8, color="#2c3e50")

    ax.set_xticks(x); ax.set_xticklabels(gamma_labels)
    ax.set_xlabel("Discount factor γ", fontsize=11)
    ax.set_ylabel("Iterations to convergence", fontsize=11)
    ax.set_title("VI vs PI: Iteration Count across γ values\n"
                 "(FrozenLake-v1, 4×4, slippery)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_yscale("log")  
    ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def find_policy_emergence(P, n_states, n_actions, gamma=0.99, theta=1e-4):
    print("\n── Policy Emergence (γ=0.99) " + "─" * 35)

    V_star, pi_star, total_iters, history = value_iteration_instrumented(
        P, n_states, n_actions, gamma, theta, store_iterates=True)

    print(f"VI converged in {total_iters} total iterations.")

    k_star = None
    for k, (V_k, pi_k) in enumerate(history, start=1):
        if np.array_equal(pi_k, pi_star):
            k_star = k
            V_at_kstar = V_k
            break

    if k_star is None:
        k_star = total_iters
        V_at_kstar = history[-1][0]

    gap = np.max(np.abs(V_at_kstar - V_star))

    print(f"\n  k*  =  {k_star}   (policy stabilised here)")
    print(f"  Total VI iterations  =  {total_iters}")
    print(f"  ‖V_{{k*}} − V*‖_∞  =  {gap:.6f}")
    print(f"\n  Policy emerged after {k_star / total_iters * 100:.1f}% "
          f"of the value-convergence iterations.")

    return k_star, V_at_kstar, V_star, gap, total_iters, history, pi_star


def plot_emergence(history, k_star, V_star, total_iters,
                   save_path="policy_emergence.png"):
    errors = [np.max(np.abs(V_k - V_star)) for (V_k, _) in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, total_iters + 1), errors,
                color="#3498db", linewidth=1.5, label="‖$V_k − V^*$‖_∞")
    ax.axvline(k_star, color="#e74c3c", linestyle="--", linewidth=1.8,
               label=f"k* = {k_star}  (policy stabilises)")
    ax.set_xlabel("VI iteration k", fontsize=11)
    ax.set_ylabel("‖$V_k − V^*$‖_∞  (log scale)", fontsize=11)
    ax.set_title("Policy Emergence: value error vs. iteration\n"
                 "(FrozenLake-v1, γ=0.99)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def print_summary_table(results, gammas):
    print("\n" + "=" * 75)
    print(f"{'γ':>7}  {'VI iters':>9}  {'VI time(ms)':>11}  "
          f"{'VI backups':>13}  {'PI iters':>9}  {'PI time(ms)':>11}  "
          f"{'PI backups':>13}")
    print("=" * 75)
    for g in gammas:
        vi = results[g]["vi"]
        pi = results[g]["pi"]
        print(f"{g:>7}  {vi['iters']:>9d}  {vi['time']*1000:>11.2f}  "
              f"{vi['backups']:>13,}  {pi['iters']:>9d}  "
              f"{pi['time']*1000:>11.2f}  {pi['backups']:>13,}")
    print("=" * 75)


if __name__ == "__main__":
    
    GAMMAS = [0.5, 0.9, 0.99, 0.999]
    THETA  = 1e-4
    P, n_states, n_actions, map_grid, env = make_env()

#   ───────────────── 3(c): Empirical comparison ─────────────────
    results = run_comparison(P, n_states, n_actions, GAMMAS, theta=THETA)
    print_summary_table(results, GAMMAS)
    plot_comparison(results, GAMMAS)

#    ───────────────── 3(d): Policy emergence ─────────────────
    (k_star, V_at_kstar, V_star, gap,
     total_iters, history, pi_star) = find_policy_emergence(
        P, n_states, n_actions, gamma=0.99, theta=THETA)

    print("\nV* (converged):")
    print_value_table(V_star, label="V*")
    print(f"\nV_{{k*}} at k*={k_star}:")
    print_value_table(V_at_kstar, label=f"V_{{k*={k_star}}}")

    print(f"\nπ* (optimal policy):")
    print_policy(pi_star, map_grid)

    plot_emergence(history, k_star, V_star, total_iters)

    env.close()