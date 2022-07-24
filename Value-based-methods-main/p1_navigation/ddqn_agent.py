import tensorflow as tf
import numpy      as np

import random , sys , threading

from collections import deque
from dmodel      import QNetwork , Variables , lambda_function

settings = Variables ()

class Agent () :

    def __init__ ( self , state_size , action_size , seed ) :
        global            settings
        self.settings   = settings
        self.settings.set_vars ( state_size , action_size , seed )

        self.qnetwork_target = QNetwork ( settings )
        self.qnetwork_local  = QNetwork ( settings )

        self.memory = ReplayBuffer ( settings )
        self.t_step = 0
        self.last_action = 0
        self.last_state  = 0

    # ---------------------------------------------------------------------------

    def act ( self , state , eps = 0. ) :
        action_values = self.qnetwork_local.predict ( state , 0 )

        if random.random () > eps : self.last_action = np.argmax     (                           action_values )
        else:                       self.last_action = random.choice ( np.arange ( self.settings.action_size   ) )
        return self.last_action

    # ---------------------------------------------------------------------------

    def step ( self     , state , action , reward , next_state , done ) :
        self.memory.add ( state , action , reward , next_state , done )
        self.last_state =                           next_state

        self.t_step = ( self.t_step + 1 )                 % self.settings.UPDATE_EVERY
        if ( self.t_step == 0 ) and ( len ( self.memory ) > self.settings.BATCH_SIZE  ) :
             self.learn (                   self.memory.sample ()                     )

    # ---------------------------------------------------------------------------

    # experience from memory
    def learn ( self , experiences ) :
        states , actions , rewards , next_states , dones = experiences

        # ------------------------------------------------------------------------------------------------------|
        # we can't predict using a generic states predictions because other actions must still not predicted    |
        # we can't predict states' actions one by one because we need global loss for Adam optimization         |
        # we can't use memory for the previously predictions because they are like a reinforcement for actions  |
        # ------------------------------------------------------------------------------------------------------|

        predicted_nextstates = self.qnetwork_target.predict_series ( next_states , 0 )
        predicted_states     = self.qnetwork_local. predict_series (      states , 0 )

        len_states  =    len   (     states )
        predictions = [ None ] * len_states

        print ( '' )
        for i , next , state in zip ( range ( len_states ) , predicted_nextstates , predicted_states ) :
            max_value = max ( next )
            max_value = rewards [ i ] + ( self.settings.GAMMA * max_value * ( 1 - dones [ i ] ) )
            state     [ actions [ i ] ] = max_value
            predictions         [ i ]   = state

        # ------------------------------------------------------------------------------------------------------

        predictions = np.vstack  ( predictions )
        assert len_states == len ( predictions )

        self.qnetwork_local.training ( states , predictions , 0 )
        print ( '\nLoss:' , int ( self.settings.loss ) , 'Accuracy:' , round ( self.settings.accuracy , 2 ) )

        # loss = F.mse_loss(Q_expected, Q_targets) # == mse as loss
        # self.optimizer.zero_grad ()              # == None as gradients = RNN = short-memory
        # loss.backward            ()              # == using a different loss from another random input, Double-QNN
        # self.optimizer.step      ()              # == run the training ( derivates, back-propgation=RNN, Adam )
        # back-propagation -> so RNN

        # -----------------------------------------------------------------------

        target_weights = [ ( 1.0 - settings.TAU ) * el for el in self.qnetwork_target.model.trainable_weights ]
        local_weights  = [         settings.TAU   * el for el in self.qnetwork_local. model.trainable_weights ]

        print ( 'Network Target is updating\n' )

        weights = [ el1 + el2 for el1 , el2 in zip ( target_weights , local_weights ) ]
        self.qnetwork_target.setWeights            (                        weights )

# ---------------------------------------------------------------------------

class ReplayBuffer () :

    def __init__ ( self , settings ) :
        self.settings   = settings

        self.memory = deque ( maxlen = settings.BUFFER_SIZE )
        self.batch_size =              settings.BATCH_SIZE
        self.seed = random.seed      ( settings.SEED        )
    
    def add ( self         ,   state , action , reward , next_state , done ) :
        self.memory.append ( ( state , action , reward , next_state , done ) )
    
    def sample ( self ) :
        experiences = random.sample ( self.memory , k = self.settings.BATCH_SIZE )
        # it will put us in a brute-force situation, any casual good situation will define our correlations
        # list could be [ 6 , 3 , 4 ] not only [ 3, 4 , 6 ]

        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        for exp in experiences :
            # print ( 'exp:' , exp )
            states      += [ exp [ 0 ] ]
            actions     += [ exp [ 1 ] ]
            rewards     += [ exp [ 2 ] ]
            next_states += [ exp [ 3 ] ]
            dones       += [ exp [ 4 ] ]

        # vstack -> list for each row
        return ( np.vstack (      states ) , np.vstack ( actions ) , np.vstack ( rewards ) ,
                 np.vstack ( next_states ) , np.vstack ( dones   ) )

    def __len__ ( self ) : return len ( self.memory )
