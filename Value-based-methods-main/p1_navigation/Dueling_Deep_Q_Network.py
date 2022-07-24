import gym
import random
import torch
import numpy as np

from collections import deque

import matplotlib.pyplot as plt

import sys , os , subprocess

sys.path.append ( '/home/kaumi/Git/deepL_RL/' )
from saveFigure import save , plot , setLab ,  plot_series

setLab ( 'Dueling_Deep_Q_Network' )

# --------------------------------------------------------
# --------------------------------------------------------

from unityagents import UnityEnvironment
env = UnityEnvironment ( file_name = './Banana_Linux/Banana.x86_64' )

brain_name = env.brain_names [ 0 ]
brain      = env.brains [ brain_name ]
env_info = env.reset ( train_mode = True ) [ brain_name ]

action_size = brain.vector_action_space_size
print ( 'Number of agents:'  , len ( env_info.agents ) )
print ( 'Number of actions:' ,           action_size ) # 4

state = env_info.vector_observations [ 0 ]
state_size =              len ( state      )
print ( 'States look like:'   , state      )
print ( 'States have length:' , state_size ) # 37

# --------------------------------------------------------
# --------------------------------------------------------

from ddqn_agent import Agent
agent = Agent ( state_size = state_size , action_size = action_size , seed = 0 )

# --------------------------------------------------------
# --------------------------------------------------------

def ddqn ( n_episodes = 2000 , max_t = 1000 , eps_start = 1.0 , eps_end = 0.01 , eps_decay = 0.995 ) :
    scores = []                            # list containing scores from each episode
    scores_window = deque ( maxlen = 100 ) # last 100 scores
    eps = eps_start                        # initialize epsilon

    for i_episode in range   ( 1 , n_episodes + 1 ) :
        env_info = env.reset ( train_mode = True  ) [ brain_name ]
        state    = env_info.vector_observations [ 0 ]
        score    = 0
        
        # print ( 'starting state:' , state )

        for t in range ( max_t ) :
            print ( '\rEpisode: {} in try: {}'.format ( i_episode , t ) , end = '' )
            action = agent.act ( state , eps )

            env_info   = env.step ( action ) [ brain_name ]            
            next_state = env_info.vector_observations [ 0 ]
            reward     = env_info.rewards [ 0 ]
            done       = env_info.local_done [ 0 ]
            score     += reward
            state      = next_state
            score     += reward

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

scores = ddqn ()
print ( 'scores:' , scores )

# --------------------------------------------------------

# 5. Explore
## - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster. 
##   Once you build intuition for the hyperparameters that work well with this environment,
##   try solving a different OpenAI Gym task with discrete actions!
## - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN! 
## - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.  
