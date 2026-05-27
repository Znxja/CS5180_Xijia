from env import ApartmentEnv
from policies import RandomPolicy

def main():
    T, K = 4, 4
    env = ApartmentEnv(T,K)
    policy = RandomPolicy(T)

    obs, info = env.reset()
    done = False
    print(f'{'Step':>4}  {'t':>2}  {'U_t':>3}  {'Action':>8}  {'Reward':>6}  {'Done':>5}')
    print('-'*45)

    step = 0
    while not done:
        step += 1
        t = obs['t']
        u = obs['U_t']
        action = policy.act(obs)
        action_str = 'ACCEPT' if action == 1 else 'REJECT'
        obs, reward, terminated, truncated, info =  env.step(action)
        done = terminated or truncated

        print(f'{step:>4} {t:>2} {u:>3} {action_str:>8} {reward:>6} {str(done):>5}')

    if reward > 0:
        print(f'Accepted apartment with quality of {reward} at week {t}')
    else:
        print(f'Rejected all apartments, utility = 0')

if __name__ == '__main__':
    main()