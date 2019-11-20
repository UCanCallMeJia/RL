'''

使用DQN 用于 连续控制任务
将-2~2 的action 离散化
使用DQN 算法

'''


import gym
import numpy as np 
from RL_brain_Pendulum import DeepQNetwork
env = gym.make('Pendulum-v0')
env = env.unwrapped

# print(env.action_space)             # a number
# print(env.observation_space)        # np.array([np.cos(theta), np.sin(theta), thetadot])   
# print(env.observation_space.high)   
# print(env.observation_space.low)
# Box(1,)
# Box(3,)
# [1. 1. 8.]
# [-1. -1. -8.]

action_space = np.linspace(-2.,2.,num=400)
# print(action_space)

RL = DeepQNetwork(
            n_actions=len(action_space),
            n_features=env.observation_space.shape[0],
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=300,
            batch_size=50,
            #e_greedy_increment=0.001
)

total_step = 0

for i_epilson in range(100):
    
    observation = env.reset()
    # print(observation)
    # input()
    ep_reward = 0
    steps = 0
    while(True):
        env.render()

        action = RL.choose_action(observation)
        # print(action)

        observation_ , reward , done , info = env.step(action)
        # if observation_[0] > 0.87:
        #     reward = reward/8 + 5
        # else:
        #     reward = reward/8 - 1
        reward = (reward+ 8)/10.
        # print(observation_) 
        # print(reward/10.0)
        # print(done)
        # input()

        RL.store_transition(observation,action,reward,observation_)

        # reward in a episode
        ep_reward += reward

        if total_step>1000:
            RL.learn()

        if steps == 1200:
            print('episode: ', i_epilson,'ep_r: ', round(ep_reward, 2),
			' epsilon: ', round(RL.epsilon, 2))
            break 
        steps+=1
        observation =observation_
        total_step += 1

RL.plot_cost()