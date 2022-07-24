import tensorflow as tf
import numpy      as np

import random , sys , threading

from collections import deque
from model       import QNetwork1 , QNetwork2 , Variables , lambda_function

settings = Variables ()

class Agent () :

    def __init__ ( self , state_size , action_size , seed ) :
        global            settings
        self.settings   = settings
        self.settings.set_vars ( state_size , action_size , seed )

        self.qnetwork_target = QNetwork1 ( settings )
        self.qnetwork_local  = QNetwork1 ( settings )
        self.predicti_local  = QNetwork2 ( settings )
        self.predicti_target = QNetwork2 ( settings )

        self.memory = ReplayBuffer ( settings )
        self.t_step = 0
        self.last_action = 0
        self.last_state  = 0

    # ---------------------------------------------------------------------------

    def act ( self , state , eps = 0. ) :
        action_values = self.predicti_local.predict ( state )

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

        def learn_predict ( predict_function , next_state , reward , GAMMA , done , array , index ) :

            max_target_prediction = predict_function ( next_state , 0 )
            max_target_prediction = np.max ( max_target_prediction )

            array [ index ] = reward + ( self.settings.GAMMA * max_target_prediction * ( 1 - done ) )

        states , actions , rewards , next_states , dones = experiences

        # ------------------------------------------------------------------------------------------------------|
        # we can't predict using a generic states predictions because other actions must still not predicted    |
        # we can't predict states' actions one by one because we need global loss for Adam optimization         |
        # we can't use memory for the previously predictions because they are like a reinforcement for actions  |
        # ------------------------------------------------------------------------------------------------------|

        len_states = len ( states )
        predictions = [ None ] * len_states
        threads     = [ None ] * len_states

        print ( '' )
        for i in range ( len_states ) :
            threads [ i ] = threading.Thread ( target = learn_predict ,
                                               args   = ( self.predicti_target.predict ,
                                                          next_states [ i ] , rewards [ i ] , self.settings.GAMMA , dones [ i ] , predictions , i ) )
            threads [ i ].start ()
            print ( '\rmax_target_prediction progress %:' , int ( ( i + 1 ) / len ( states ) * 100 ) , end = '' )

        for T in threads : T.join ()

        # ------------------------------------------------------------------------------------------------------

        print ( '.\n' )
        print ( 'len states:' , len_states )

        predictions = np.vstack  ( predictions )
        assert len_states == len ( predictions )

        print ( '' )
        sum_loss = 0

        for i in range ( len ( states ) ) :
            action =           actions [ i ]
            action =     int ( action )

            self.qnetwork_local.setWeights ( self.qnetwork_local.model.weights [ : -2 ] + self.settings.weights [ action ] )

            loss = int ( self.qnetwork_local.training ( np.vstack ( [ states      [ i ] ] ) ,
                                                        np.vstack ( [ predictions [ i ] ] ) , 0 ) )
            sum_loss +=                         loss
            print (  '\rSum Loss:'  ,       sum_loss ,
                       'Mean Loss:' , int ( sum_loss / ( i + 1 ) ) , '%:' , int ( ( i + 1 ) / len ( states ) * 100 ) , end = ' . ' )

            self.settings.set_weights_i ( action , self.qnetwork_local.model.weights [ -2 : ] )

            # ------------------------------------------------------------------------------------------------------

            if ( i + 1 ) == len ( states ) :
               print ( '\n\nLast state for training reached on sample:' , i + 1 )

               predicti_local_weights          = self.predicti_local.model.weights
               predicti_local_weights [ : -2 ] = self.qnetwork_local.model.weights [ : -2 ]

               print ( 'predicti_local_weights final shape 2:' , predicti_local_weights [ -2 ].shape )
               print ( 'predicti_local_weights final shape 1:' , predicti_local_weights [ -1 ].shape )

               for u in range ( self.settings.action_size ) :

                   tmp_array_2 = np.array ( self.settings.weights [ u ][ -2 ] )
                   tmp_array_1 = np.array ( self.settings.weights [ u ][ -1 ] )

                   if u > 0:
                            predicti_local_weights [ -2 ] = np.concatenate ( ( predicti_local_weights [ -2 ] , tmp_array_2 ) , axis=1 )
                            predicti_local_weights [ -1 ] = np.concatenate ( ( predicti_local_weights [ -1 ] , tmp_array_1 ) , axis=0 )
                   else :
                            predicti_local_weights [ -2 ] = tmp_array_2
                            predicti_local_weights [ -1 ] = tmp_array_1

                   print ( 'Level' , u + 1 , 'shape predicti_local_weights 2:' , predicti_local_weights [ -2 ].shape )
                   print ( 'Level' , u + 1 , 'shape predicti_local_weights 1:' , predicti_local_weights [ -1 ].shape )

               self.predicti_local.setWeights ( predicti_local_weights )

        # loss = F.mse_loss(Q_expected, Q_targets) # == mse as loss
        # self.optimizer.zero_grad ()              # == None as gradients
        # loss.backward            ()              # == back-propagation RNN
        # self.optimizer.step      ()              # == run the training
        # back-propagation -> so -> Bidirectional

        # -----------------------------------------------------------------------

        target_weights = [ ( 1.0 - settings.TAU ) * el for el in self.predicti_target.model.trainable_weights ]
        local_weights  = [         settings.TAU   * el for el in self.predicti_local. model.trainable_weights ]

        print ( 'Network Target is updating' ) ; print ( '' )

        weights = [ el1 + el2 for el1 , el2 in zip ( target_weights , local_weights ) ]
        self.predicti_target.setWeights            (                        weights )

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
