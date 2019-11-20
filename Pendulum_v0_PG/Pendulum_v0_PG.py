import gym
import numpy as np
from RL_agent_PG import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 0  # renders environment if total episode reward is greater then this threshold
RENDER = True  # rendering wastes time

env = gym.make('Pendulum-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

action_space = np.linspace(-2.,2.,num=200)
action_n = len(action_space)

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=action_n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.1,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(5000):

    observation = env.reset()
    
    episode_step = 0

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        episode_step += 1

        observation_, reward, done, info = env.step(action)

        reward = reward/10.

        if action > 0:
            action_to_index = action_n/2 + int(action/(4.0/action_n))
        else:
            action_to_index = action_n/2 - int(np.abs(action)/(4.0/action_n))
        # print('index: ',action_to_index)
        RL.store_transition(observation, action_to_index, reward)

        if episode_step == 200:
            ep_rs_sum = sum(RL.ep_rs)

            if ep_rs_sum > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode,'  episode_step:',episode_step , "  reward:", int(ep_rs_sum))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
