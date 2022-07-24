import gym
import random
import torch
import numpy as np

from collections import deque

import matplotlib.pyplot as plt

import sys , os , subprocess
# print ( subprocess.check_output ( 'pip  install --upgrade pip'    , shell=True , stderr=subprocess.STDOUT ) )
# print ( subprocess.check_output ( 'pip install -U tensorflow'     , shell=True , stderr=subprocess.STDOUT ) )
# print ( subprocess.check_output ( 'pip install -U tensorflow-gpu' , shell=True , stderr=subprocess.STDOUT ) )
# print ( subprocess.check_output ( 'pip show       tensorflow'     , shell=True , stderr=subprocess.STDOUT ) )

sys.path.append ( '/home/kaumi/Git/deepL_RL/' )
from saveFigure import save , plot , setLab ,  plot_series

setLab ( 'Deep_Q_Network' )
env = gym.make ( 'LunarLander-v2' )
env.seed ( 0 )

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# --------------------------------------------------------

from dqn_agent import Agent

agent = Agent ( state_size = 8 , action_size = 4 , seed = 0 )

state = env.reset ()

# for j in range ( 200 ) :
#     action = agent.act ( state )
#     env.render ()
#     state, reward, done, _ = env.step ( action )
#     if done : break 
        
env.close ()

# --------------------------------------------------------
# Train the Agent with DQN

def dqn ( n_episodes = 2000 , max_t = 1000 , eps_start = 1.0 , eps_end = 0.01 , eps_decay = 0.995 ) :
    scores = []                            # list containing scores from each episode
    scores_window = deque ( maxlen = 100 ) # last 100 scores
    eps = eps_start                        # initialize epsilon

    for i_episode in range ( 1 , n_episodes + 1 ) :
        state = env.reset ()
        score = 0
        
        # print ( 'starting state:' , state )

        for t in range ( max_t ) :
            print ( '\rEpisode: {} in try: {}'.format ( i_episode , t ) , end = '' )
            action = agent.act ( state , eps )
            next_state , reward , done , _ = env.step ( action )
            agent.step ( state , action , reward , next_state , done )
            state = next_state
            score += reward

            if done : break

        scores_window.append ( score ) # save most recent score
        scores.append        ( score ) # save most recent score

        eps = max ( eps_end , eps_decay * eps ) # decrease epsilon

        print ( '\n\nEpisode {} Average Score: {:.2f}\n'.format ( i_episode , np.mean ( scores_window ) ) )

        if np.mean ( scores_window ) >= 200.0 : 
           print ( 'Environment solved in {:d} episodes! Average Score: {:.2f}'.format ( i_episode - 100 , np.mean ( scores_window ) ) )
           # torch.save ( agent.qnetwork_local.state_dict () , 'checkpoint.pth' )
           break

    return scores

scores = dqn ()
print ( 'scores:' , scores )

# plot the scores
# fig = plt.figure ()
# ax  = fig.add_subplot ( 111 )
# plt.plot ( np.arange ( len ( scores ) ) , scores )
# plt.ylabel ( 'Score'     )
# plt.xlabel ( 'Episode #' )
# plt.show ()

# --------------------------------------------------------
# 4. Watch a Smart Agent!

# load the weights from file
# agent.qnetwork_local.load_state_dict ( torch.load ( 'checkpoint.pth' ) )

# for i in range ( 5 ) :
#    state = env.reset ()

#    for j in range ( 200 ) :
#        action = agent.act ( state )
#        env.render ()
#        state, reward, done, _ = env.step ( action )

#        if done : break 
            
# env.close ()

# --------------------------------------------------------

# 5. Explore
## - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster. 
##   Once you build intuition for the hyperparameters that work well with this environment,
##   try solving a different OpenAI Gym task with discrete actions!
## - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN! 
## - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.  
