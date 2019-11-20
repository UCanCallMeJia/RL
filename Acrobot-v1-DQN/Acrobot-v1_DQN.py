import gym
from RL_agent_DQN import DeepQNetwork

env = gym.make('Acrobot-v1')
env = env.unwrapped
ifrender = False

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high) 
# print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=30, memory_size=300,
                  e_greedy_increment=None,)

total_steps = 0


for i_episode in range(1000):

    observation = env.reset()

    ep_r = 0
    while True:
        if ifrender: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        cos0, sin0 , cos1 , sin1 , s0 , s1 = observation_

        if reward == 0:
            reward = 5
        else:
            reward = -1

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 500 and total_steps%10 == 0:
            RL.learn()

        if done:
            if ep_r > -100:
                ifrender = True

            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
