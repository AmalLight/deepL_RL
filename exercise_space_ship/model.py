import tensorflow as tf
import numpy      as np

# -----------------------------------------------------

class Variables () :

    def __init__ ( self  , state_size = 0 , action_size = 0 , seed = 0 ) :
        self.state_size  = state_size
        self.action_size = action_size
        self.loss        = 0
        self.accuracy    = 0

        self.BUFFER_SIZE = int ( 1e5 ) # replay buffer size
        self.BATCH_SIZE = 64           # minibatch size
        self.GAMMA = 0.99              # discount factor
        self.TAU = 1e-3                # for soft update of target parameters
        self.LR = 5e-4                 # learning rate
        self.UPDATE_EVERY = 4          # how often to update the network
        self.EPOCHES = 1
        self.SEED = seed

    def set_vars ( self  , state_size , action_size , seed ) :
        self.state_size  = state_size
        self.action_size = action_size
        self.SEED        = seed

# -----------------------------------------------------

class lambda_function () :

    def __init__ ( self , settings ) :
        self.settings   = settings

    def expand ( self , x ) :
        return tf.expand_dims ( x , axis = -1 )
        # LSTM requires this option, x are states

    #   saved weights
    def saved ( self , x ) :
        return x

# -----------------------------------------------------

class QNetwork () :

    def __init__ ( self    , settings , deep = 64 ) :
        self.settings      = settings
        self.object_lambda = lambda_function ( settings )

        # self.optimizer = tf.keras.optimizers.Adam ( learning_rate = settings.LR )
        self.optimizer   = tf.keras.optimizers.Adam ()

        # Flatten is necessary for multi-dimension ( Matrix )
        # activation = ( tf.nn.relu => if None == nn.Linear in torch )
        # here tf.nn.relu is requested

        # if go_backwards : inputs = reverse ( inputs , 0 )
        # go_backwards => first states will have the values of latest
        # SimpleRNN == RNN + Dense != LSTM for long memory ==> it is only for short memory

        self.shape  = (deep,)
        self.shapeX =  deep

        # inside the first unit there are 8 columns == 8 variables
        # -------------------------------------------------------------------------------------------
        self.model =     tf.keras.Sequential    ()
        self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'DenseInput' ) )

        self.model.add ( tf.keras.layers.SimpleRNN ( deep , activation = tf.nn.relu , name = 'DenseDeep1' , return_sequences = True  , go_backwards = True  ) )
        self.model.add ( tf.keras.layers.SimpleRNN ( deep , activation = tf.nn.relu , name = 'DenseDeep2' , return_sequences = False , go_backwards = False ) )

        self.model.add ( tf.keras.layers.Dense  ( settings.action_size , activation = None , name = 'DenseOutput' ) )
        self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.saved                 , name = 'DenseSave'   ) )
        # -------------------------------------------------------------------------------------------

        self.model.compile ( optimizer = self.optimizer , loss = 'mean_squared_error' , metrics = [ 'mse' , 'mae' , 'accuracy' ] )
        self.model.summary ()

    # ----------------------------------------------------------------------

    def training ( self           , states , labels ,                                  verbose = 1       ) :
        result   = self.model.fit ( states , labels , epochs = self.settings.EPOCHES , verbose = verbose )
        self.settings.loss , self.settings.accuracy = result.history [ 'loss' ][ -1 ] , result.history [ 'accuracy' ][ -1 ]

    def predict ( self               , state                                              , verbose = 1       ) :
        return    self.model.predict ( state.reshape ( ( 1 , self.settings.state_size ) ) , verbose = verbose ) [ 0 ]

    def predict_series ( self               , states , verbose = 1       ) :
        return           self.model.predict ( states , verbose = verbose )

    def setWeights ( self , weights ) : self.model.set_weights ( weights )
