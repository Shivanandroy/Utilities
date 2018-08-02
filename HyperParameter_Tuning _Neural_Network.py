# importing libraries
!pip install --upgrade git+git://github.com/hyperopt/hyperopt.git
import keras
from keras.utils import np_utils
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
import sys
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.copy()
X_test = x_test.copy()


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


X = X_train.copy()
y = y_train.copy()
X_val = X_test.copy()
y_val = y_test.copy()

space = { 'choice': hp.choice('layers_number',
                             [{'layers': 'two'},
                             {'layers': 'three',
                             'units3': hp.choice('units3', [32, 64, 256]),
                             'dropout3': hp.choice('dropout3', np.linspace(0.1, 0.3, 3, dtype=float))
                             }]),

            'units1': hp.choice('units1', [512, 768, 1024]),
            'units2': hp.choice('units2', [128, 256, 512]),
            #'units3': hp.choice('units3', [32, 64, 256]), 

            'dropout1': hp.choice('dropout1', np.linspace(0.3, 0.5, 3, dtype=float)),
            'dropout2': hp.choice('dropout2', np.linspace(0.1, 0.3, 3, dtype=float)),
            #'dropout3': hp.choice('dropout3', np.linspace(0.1, 0.3, 3, dtype=float)),

            'batch_size' : hp.choice('batch_size', [128, 256, 512]),

            'nb_epochs' :  hp.choice('nb_epochs', [30, 50, 100]),
           'activation': 'relu',
         'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop'])
            
        }

def f_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'])

    model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 3)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = accuracy_score(y_val, pred_auc.round())
    print('AUC:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print ('best: ')
print (best)