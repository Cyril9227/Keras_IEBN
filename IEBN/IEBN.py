import keras.backend as K
from keras import layers
from keras.initializers import Constant

class BatchAttNorm(layers.BatchNormalization):
    def __init__(self, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(layers.BatchNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon  = epsilon
        self.axis = axis
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input_shape))
            
        dim = input_shape[self.axis]
        shape = (dim, )
    
        self.weight = self.add_weight(name='weight', 
                                      shape=shape,
                                      initializer=Constant(1),
                                      trainable=True)

        self.bias = self.add_weight(name='bias', 
                                    shape=shape,
                                    initializer=Constant(0),
                                    trainable=True)

        self.weight_readjust = self.add_weight(name='weight_readjust', 
                                               shape=shape,
                                               initializer=Constant(0),
                                               trainable=True)
        
        self.bias_readjust = self.add_weight(name='bias_readjust', 
                                             shape=shape,
                                             initializer=Constant(-1),
                                             trainable=True)
        

        super(layers.BatchNormalization, self).build(input_shape)


    def call(self, input):
        if self.axis == -1:
            data_format = 'channels_last'
        else:
            data_format = 'channel_first'
            
        avg = layers.GlobalAveragePooling2D(data_format)(input) 
        attention = K.sigmoid(avg * self.weight_readjust + self.bias_readjust)

        bn_weights = self.weight * attention          
        
        
        out_bn = layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon, 
                                           axis=self.axis, center=False, scale=False)(input)
        
        if K.int_shape(input)[0] is None or K.int_shape(input)[0] > 1:
            bn_weights = bn_weights[:, None, None, :]
            self.bias  = self.bias[None, None, None, :]
            
        out_bn = out_bn * bn_weights + self.bias
 
        return out_bn

    def get_config(self):
        config = {
            'axis' : self.axis,
            'momentum' : self.momentum,
            'epsilon' : self.epsilon
        }
        base_config = super(layers.BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
