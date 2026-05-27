import numpy as np
import matplotlib.pyplot as plt
from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy

def run_episodes(env_kwargs, policy, N=10000, seed=0):
    res = np.zeros(N)
    for i in range(N):
        env = ApartmentEnv(**env_kwargs,seed = seed+i)
        obs, info = env.reset()
        done = False
        episode_res = 0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_res += reward
            done = terminated or truncated
        res[i] = episode_res
    return res

def episode_stas(name,res):
    mean = np.mean(res)
    se = np.std(res,ddof=1)/np.sqrt(len(res))
    rej_frac = np.mean(res == 0)
    print(f'{name}, mean = {mean:.4f} +- {se:.4f}, fraction_rejected_all = {rej_frac:.4f}')
    return mean, se, rej_frac

T, K= 4, 4
env_kwargs = {"T": T, "K": K}
policies = {
    "RandomPolicy": RandomPolicy(T),
    "ThresholdPolicy(u_min=3)": ThresholdPolicy(3),
    "OptimalPolicy": OptimalPolicy(T, K),
}

results = {}
for name, policy in policies.items():
    res = run_episodes(env_kwargs, policy)
    mean, se, rej_frac = episode_stas(name, res)
    results[name] = res

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
colors = ["tab:blue", "tab:orange", "tab:green"]
bins = np.arange(-0.5, K + 1.5, 1)  
for ax, (name, res), color in zip(axes, results.items(), colors):
    ax.hist(res, bins=bins, color=color, edgecolor="white", density=True)
    ax.set_title(f"{name}\nmean={np.mean(res):.3f}")
    ax.set_xlabel("Return (utility)")
    ax.set_xticks(range(0, K + 1))

axes[0].set_ylabel("Density")
fig.suptitle("Return distributions: T=4, K=4, N=10,000", fontweight="bold")
plt.tight_layout() 
# plt.savefig("histogram.png", dpi=150)
# plt.show()

thres_res = {}
for u in range(1,5):
    name = f'ThresholdPolicy(u_min={u})'
    policy = ThresholdPolicy(u)
    res = run_episodes(env_kwargs,policy)
    mean, se, rej_frac = episode_stas(name,res)
    thres_res[u] = {'mean':mean, 'se':se, 'rej_frac':rej_frac, 'res':res}
best_thres = max(thres_res, key = lambda x: thres_res[x]['mean'] )
print(f'Best-performance threshold: {best_thres}, (mean = {thres_res[best_thres]['mean']:.4f} +- {thres_res[best_thres]['se']:.4f})')

sigmas = [0, 0.5, 1, 2]
noise_res = {name: [] for name in policies}
for sigma in sigmas:
    env_kwargs_noise = {"T": T, "K": K, 'noise_std':sigma}
    print(f'noise_std: {sigma}')
    for name, policy in policies.items():
        res = run_episodes(env_kwargs_noise,policy)
        mean, se, rej_frac = episode_stas(name, res)
        noise_res[name].append(mean)

for name, means in noise_res.items():
    drop = means[-1] - means[0]
    pct = drop/means[0] * 100 if means[0] != 0 else 0
    print(f'{name}: noise_std=0 mean={means[0]:.3f}, noise_std=2 mean={means[-1]:.3f}, drop={drop:.3f} ({pct:.1f}%)')