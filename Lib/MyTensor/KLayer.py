from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization, UpSampling1D, Conv2D

def CNN1DLayer(Input,filters,fshape,mpooling=True,DropOut=True,BatchNorm=False,Padding='SAME',activation='relu'):
    conv = Conv1D(filters,fshape,padding=Padding,activation=activation)(Input)
    layer = conv
    if(mpooling):
        maxp = MaxPooling1D(2,padding=Padding)(layer)
        layer = maxp
    if(Dropout):
        drop = Dropout(0.5)(layer)
        layer = drop
    if(BatchNorm):
        batch = BatchNormalization()(layer)
        layer = batch
    return layer

def CNN2DLayer(Input,filters,fshape,mpooling=True,DropOut=True,BatchNorm=False,Padding='SAME',activation='relu'):
    conv = Conv2D(filters,fshape,padding=Padding,activation=activation)(Input)
    layer = conv
    if(mpooling):
        maxp = MaxPooling1D(2,padding=Padding)(layer)
        layer = maxp
    if(Dropout):
        drop = Dropout(0.5)(layer)
        layer = drop
    if(BatchNorm):
        batch = BatchNormalization()(layer)
        layer = batch
    return layer

def DecodeLayer(Input,filters,fshape,DropOut=True,UpSamp=True,BatchNorm=False,activation='relu'):
    de_conv = Conv1D(filters,fshape,padding='SAME',activation=activation)(Input)
    de_layer = de_conv
    if(UpSamp):
        de_max = UpSampling1D()(de_layer)
        de_layer = de_max
    if(Dropout):
        de_drop = Dropout(0.5)(de_layer)
        de_layer = de_drop
    if(BatchNorm):
        de_batch = BatchNormalization()(de_layer)
        de_layer = de_batch
    return de_layer
	
def LSTMLayer(Input,lstm_dim,returnSequences=False,DropOut=True,BatchNorm=False):
    lstm = LSTM(lstm_dim,return_sequences=returnSequences)(Input)
    layer = lstm
    if(Dropout):
        drop = Dropout(0.5)(layer)
        layer = drop
    if(BatchNorm):
        batch = BatchNormalization()(layer)
        layer = batch
    return layer