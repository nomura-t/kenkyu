import tensorflow
import math
import tflearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import metrics



tensorflow.reset_default_graph()
json_input = []
json_output = []
tweets = []
for i in range(80):
	json_input.append(json.load(open('./result/result'+str(i*3)+'.txt')))
	json_output.append(json.load(open('./result/result'+str(i*3+2)+'.txt')))
	tweets.append(np.array(pickle.load(open('./vector/vector'+str(i)+'.txt','rb'))))
tweets = np.array(tweets)
print(tweets[0].shape)
tweets1 =[]
for i in range(80):
	tweets1.append(tweets[i][0:1200,:])
inputdata = []
outputdata = []
Y_Openness = []
Y_Conscientiousness = []
Y_Extraversion = []
Y_Agreeableness = []
Y_Emotional_range = []
for s in range(80):
	kariin = []
	kariout =[]
	for i in range(5):
		kariin.append(json_input[s]['personality'][i]['percentile'])
		kariout.append(json_output[s]['personality'][i]['percentile'])
	inputdata.append(kariin)
	outputdata.append(kariout)
#X = np.array(inputdata).reshape(80,-1,5)
#Y = np.array(outputdata).reshape(80,-1,5)
X = np.array(inputdata)
Y = np.array(outputdata)
print(X.shape)
Y = X - Y
with open('dltbig5.pkl','wb') as f:
	pickle.dump(Y,f)
for i in range(80):
	Y_Openness.append(Y[i][0])
	Y_Conscientiousness.append(Y[i][1])
	Y_Extraversion.append(Y[i][2])
	Y_Agreeableness.append(Y[i][3])
	Y_Emotional_range.append(Y[i][4])

X_tweets = np.array(tweets1)
#X_train = X[0:65]
Y_train = Y[0:65]
Y_Openness_train = Y_Openness[0:65]
Y_Conscientiousness_train = Y_Conscientiousness[0:65]
Y_Extraversion_train = Y_Extraversion[0:65]
Y_Agreeableness_train = Y_Agreeableness[0:65]
Y_Emotional_range_train = Y_Emotional_range[0:65]
X_tweets_train = X_tweets[0:65]
#X_test = X[65:]
#testdata

Y_test = Y[65:]
Y_Openness_test = Y_Openness[65:]
Y_Conscientiousness_test = Y_Conscientiousness[65:]
Y_Extraversion_test = Y_Extraversion[65:]
Y_Agreeableness_test = Y_Agreeableness[65:]
Y_Emotional_range_test = Y_Emotional_range[65:]
X_tweets_test = X_tweets[65:]

#活性化関数の指定

activation = 'sigmoid'

def get_model(input_shape):
	#model = ['LSTM',Sequential([LSTM(100,input_shape=input_shape),Dense(5),Activation("linear")])]
	model = ['LSTM',Sequential([LSTM(100,input_shape=input_shape),Dense(1),Activation(activation)])]
	return model[1]



#入力の形状
input_shape=(1200,300)
model = get_model(input_shape)

#最適化手法の設定
opt = optimizers.Adam()

#modelの実行
model.compile(optimizer = opt,loss = 'mean_squared_error',metrics = ['accuracy'])
model.summary()
temp1 = [Y_Openness_train,Y_Conscientiousness_train,Y_Extraversion_train,Y_Agreeableness_train,Y_Emotional_range_train]
temp2 = [Y_Openness_test,Y_Conscientiousness_test,Y_Extraversion_test,Y_Agreeableness_test,Y_Emotional_range_test]
temp_name = ['Openness','Conscientiousness','Extraversion','Agreeableness','Emotional_range']

#ハイパーパラメタ
numnum = 3
Y_temp_train = temp1[numnum]
Y_temp = temp2[numnum]
epoch = 11
batch_si = 65
history = model.fit(X_tweets_train,Y_temp_train,epochs=epoch,batch_size=batch_si,validation_split=0.1)
#history = model.fit(X_tweets_train,Y_train,epochs=8,batch_size=10,validation_split=0.1)
#history = model.fit(X_tweets_train,Y_Openness_train,epochs=8,batch_size=65,validation_split=0.1)
#history = model.fit(X_tweets_train,Y_Conscientiousness_train,epochs=8,batch_size=10,validation_split=0.1)
#history = model.fit(X_tweets_train,Y_Extraversion_train,epochs=8,batch_size=10,validation_split=0.1)
#history = model.fit(X_tweets_train,Y_Agreeableness_train,epochs=8,batch_size=10,validation_split=0.1)
#history = model.fit(X_tweets_train,Y_Emotional_range_train,epochs=8,batch_size=10,validation_split=0.1)
score =[]
kari = []
heikin_t=[]
heikin_r=[]
for i in range(0,len(Y_test)):
	y_ =model.predict(X_tweets_test[i:i+1, :, :])
	#arr = np.random.rand(5)
	#arr = np.zeros(5)
	arr = np.zeros(1)
	#arr_random = np.zeros(1)
	#print(y_,Y_test[i])
	#print('t',np.sum((y_-Y_test[i])**2))
	#print('r',np.sum((arr-Y_test[i])**2))
	#print('t',np.sum((y_-Y_Openness_test[i])**2))
	#print('r',np.sum((arr-Y_Openness_test[i])**2))
	#print('t',((y_-Y_Openness_test[i])**2))
	#print('t',np.sum((y_-Y_Conscientiousness_test[i])**2))
	#print('r',np.sum((arr-Y_Conscientiousness_test[i])**2))
	#print('t',np.sum((y_-Y_Extraversion_test[i])**2))
	#print('r',np.sum((arr-Y_Extraversion_test[i])**2))
	#print('t',np.sum((y_-Y_temp[i])**2))
	#print('r',np.sum((arr-Y_temp[i])**2))
	#print('t',np.sum((y_-Y_Emotional_range_test[i])**2))
	#print('r',np.sum((arr-Y_Emotional_range_test[i])**2))
	heikin_t.append(math.sqrt((y_-Y_temp[i])**2))
	heikin_r.append(math.sqrt((arr-Y_temp[i])**2))
	#heikin_t.append((y_-Y_Conscientiousness_test[i])**2)
	#heikin_r.append((arr-Y_Conscientiousness_test[i])**2)
	#heikin_t.append((y_-Y_Extraversion_test[i])**2)
	#heikin_r.append((arr-Y_Extraversion_test[i])**2)
	#heikin_t.append((y_-Y_Agreeableness_test[i])**2)
	#heikin_r.append((arr-Y_Agreeableness_test[i])**2)
	#heikin_t.append((y_-Y_Emotional_range_test[i])**2)
	#heikin_r.append((arr-Y_Emotional_range_test[i])**2)
print('activation: '+activation)
print('epochs: '+str(epoch))
print('batch_size: '+str(batch_si))
print(temp_name[numnum])
print(sum(heikin_t)/len(Y_test))
print(sum(heikin_r)/len(Y_test))
