from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, pooling=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // (stride * pooling)

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    prev_layer = input_data
    for i in range(recur_layers):
        prev_layer = GRU(units, activation='relu', return_sequences=True, implementation=2, name=f'rnn_{i+1}')(prev_layer)
        prev_layer = BatchNormalization()(prev_layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(prev_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')
    bidir_rnn = Bidirectional(rnn, merge_mode='concat')(input_data)
    bn_rnn = BatchNormalization()(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_maxpool_rnn_model_dropout(input_dim, filters, kernel_size, conv_stride, units, conv_border_mode,
                                  cnn_layers=1, output_dim=29, dropout1=0.2, dropout2=0.2, pool_size=2):
    """ Build a recurrent + convolutional with maxpooling network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    temp_layer = input_data
    for i in range(cnn_layers):
        # Add convolutional layer
        conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name=f'conv1d_{i+1}')(temp_layer)
        # Add batch normalization
        bn_cnn = BatchNormalization(name=f'bn_conv_1d_{i+1}')(conv_1d)
        # Add dropout after convolution
        if dropout1 > 0:
            temp_layer = Dropout(dropout1, name=f'cnn_dropout_{i+1}')(bn_cnn)
        else:
            temp_layer = bn_cnn

    # Add max pooling 
    pool_cnn = MaxPooling1D(pool_size, name=f'cnn_max_pool_{i+1}')(temp_layer)
    # upsampled = UpSampling1D(size=pool_size*cnn_layers)(temp_layer)
        
    # Add a recurrent layer
    rnn = GRU(units, activation='tanh', dropout=dropout2, return_sequences=True, implementation=2, name='rnn')
    bidir_rnn = Bidirectional(rnn, merge_mode='concat')(pool_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, pooling=pool_size)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, units, conv_border_mode,
                cnn_layers=1, rnn_layers=1, output_dim=29, dropout1=0.2, dropout2=0.2, pool_size=2):
    """ Build a recurrent + convolutional with maxpooling network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    temp_layer = input_data
    for i in range(cnn_layers):
        # Add convolutional layer
        conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name=f'conv1d_{i+1}')(temp_layer)
        # Add batch normalization
        bn_cnn = BatchNormalization(name=f'bn_conv_1d_{i+1}')(conv_1d)
        # Add dropout after convolution
        if dropout1 > 0:
            temp_layer = Dropout(dropout1, name=f'cnn_dropout_{i+1}')(bn_cnn)
        else:
            temp_layer = bn_cnn

    # Add max pooling 
    pool_cnn = MaxPooling1D(pool_size, name=f'cnn_max_pool_{i+1}')(temp_layer)
      
    temp_layer = pool_cnn
    for i in range(rnn_layers):
         # Add a recurrent layer
        rnn = GRU(units, activation='tanh', dropout=dropout2, return_sequences=True, implementation=2, name=f'rnn_{i+1}')
        bidir_rnn = Bidirectional(rnn, merge_mode='concat')(temp_layer)
        # TODO: Add batch normalization
        temp_layer = BatchNormalization(name=f'bn_rnn_{i+1}')(bidir_rnn)
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(temp_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, pooling=pool_size)
    print(model.summary())
    return model