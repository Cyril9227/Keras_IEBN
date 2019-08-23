import keras.backend as K
from keras import layers
from keras.initializers import Constant

class BatchAttNorm(layers.BatchNormalization):
    def __init__(self, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(BatchAttNorm, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)
        
        if self.axis == -1:
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channel_first'
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input_shape))
                
        super(BatchAttNorm, self).build(input_shape)   
        
        dim = input_shape[self.axis]
        shape = (dim, )
        
        self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        self.GlobalAvgPooling.build(input_shape)
    
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
        

    def call(self, input):

        avg = self.GlobalAvgPooling(input) 
        attention = K.sigmoid(avg * self.weight_readjust + self.bias_readjust)

        bn_weights = self.weight * attention          
        
        out_bn = super(BatchAttNorm, self).call(input)
        
        if K.int_shape(input)[0] is None or K.int_shape(input)[0] > 1:
            bn_weights = bn_weights[:, None, None, :]
            self.bias  = self.bias[None, None, None, :]
 
        return out_bn * bn_weights + self.bias