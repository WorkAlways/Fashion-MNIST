import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(123)  
 
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import optimizers

def neural_network(epoch,batch,rate,x_train,y_train,x_dev,y_dev):
    model = Sequential()
 
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    sgd = optimizers.SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, 
          batch_size=batch, nb_epoch=epoch, verbose=1)
    score = model.evaluate(x_dev, y_dev, verbose=0)
    
    return score

def tuned_hyperparameters(dict_score):
    score_dev=[]
    for i in range(0, len(dict_score['score'])):
        score_dev.append(dict_score['score'][i][1])
    index=score_dev.index(max(score_dev))
    return dict_score['batchsize'][int(index/3)],dict_score['epochs'][int(index/10)],dict_score['learning_rate'][index]

def hyperparameter_tuning(train_x,train_y,dev_x,dev_y):
    Epochs=[10]
    batchsize=[25,50,100]
    learning_rate=[0.01,0.001,0.0001]

    dict_score={"epochs":[],
      "batchsize":[],
      "learning_rate":[],
      "score":[]}
 
    for i in range(0,len(Epochs)):
        dict_score['epochs'].append(Epochs[i])
        for j in range(0,len(batchsize)):
            dict_score['batchsize'].append(batchsize[j])
            for k in range(0,len(learning_rate)):
                dict_score['learning_rate'].append(learning_rate[k])
                dict_score['score'].append(neural_network(Epochs[i],batchsize[j],learning_rate[k],train_x,train_y,dev_x,dev_y))

    batchsize,epoch,le=tuned_hyperparameters(dict_score)
    print("Epochs",epoch)
    print("Batchsize",batchsize)
    print("Learning Rate",le)
    return(batchsize,epoch,le)

def split(df):
    return np.array(df.iloc[:,0]),np.array(df.iloc[:,1:len(df)])


def main():
    train_data=pd.read_csv('fashion-mnist_train.csv')
    test_data=pd.read_csv('fashion-mnist_test.csv')

    #Development data
    dev_data=train_data.sample(frac=0.2)
    train_data=train_data.drop(dev_data.index)
    train_data.reset_index(inplace=True, drop= True)
    dev_data.reset_index(inplace=True, drop= True)

    
    train_y,train_x=split(train_data)
    test_y,test_x=split(test_data)
    dev_y,dev_x= split(dev_data)
    
    #Preprocessing data

    rows, cols = 28, 28

    #Reshaping the data
    train_x = train_x.reshape(train_x.shape[0], rows, cols,1)
    test_x = test_x.reshape(test_x.shape[0],rows, cols,1)
    dev_x = dev_x.reshape(dev_x.shape[0], rows, cols, 1)

    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    dev_y=np_utils.to_categorical(dev_y, 10)

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    dev_x = dev_x.astype('float32')

    #Normalizing data values
    train_x /= 255
    test_x /= 255
    dev_x /= 255
    
    #Hyperparameter tuning
    #Batchsize,Epochs,Learning Rate=hyperparameter_tuning(train_x,train_y,dev_x,dev_y)

    # Epochs=10
    # Batchsize=25
    # Learning Rate=0.001

    train_accuracy=neural_network(10,25,0.001,train_x,train_y,train_x,train_y)
    dev_accuracy=neural_network(10,25,0.001,train_x,train_y,dev_x,dev_y)
    test_accuracy=neural_network(10,25,0.001,train_x,train_y,test_x,test_y)

    print("Train Accuracy",train_accuracy )
    print("Test Accuracy",train_accuracy )
    print("Development Accuracy",dev_accuracy )
    
    


    
    
    

if __name__=="__main__":
    main()
