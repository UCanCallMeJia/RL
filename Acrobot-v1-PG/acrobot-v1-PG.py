#############################
#   about after 500 episode # 
#      start to converage   #
#############################
import gym
from RL_agent_PG import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('Acrobot-v1')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()
    
    episode_step = 0

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        episode_step += 1

        observation_, reward, done, info = env.step(action)

        if reward == 0:
            reward = 5
        else:
            reward = -1

        RL.store_transition(observation, action, reward)

        if done:
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
