import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix


#data
tweets = []
#for i in range(80):
#	with open('./vector/vector'+str(i)+'.txt','rb') as f:
#		tweets.append(np.array(pickle.load(f)))
#tweets = np.array(tweets)
#maxnum = []
#for i in range(80):
#	maxnum.append(len(tweets[i]))
#maxvalue = max(maxnum)
presidentdata_x = np.full(80,0)
presidentdata_x[56:80] = 1
bigfiveparameter = 4

inputsize = 300


#for i in range(80):
#	axis2 = (maxvalue - len(tweets[i]))*300  
#	concate1 = np.concatenate(tweets[i])
#	#concate2 = np.ndarray([0]*axis2)
#	concate2 = np.full(axis2,100)	
#	tweets[i] = np.concatenate([concate1,concate2])

#with open('embedtweets.pkl','wb') as f:
#	pickle.dump(tweets,f)
#big5の値
with open('deltthreshold.pkl','rb') as f:
	deltthreshold = pickle.load(f)
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',sep=";",encoding="utf-8")
#df.head()
#
#train_x = df.drop(['quality'], axis=1)
#train_y = df['quality']
#(train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, test_size = 0.3)

train_x =[]
#BagofWordslist
with open('./BagofWords'+str(inputsize)+'.txt') as f:
	for i in range(273):
		train_x.append([int(j) for j in f.readline().split()])
merge_train_x = np.empty((inputsize,0))
for i in range(0,239,3):
	hoge = []
	for j in range(len(train_x[i])):
		hoge.append(train_x[i][j]+train_x[i+1][j]+train_x[i+2][j])
	merge_train_x = np.append(merge_train_x, np.array(hoge).reshape((-1,1)),axis=1)
	


#train_x = np.concatenate([np.array(list(tweets)),presidentdata_x.reshape(80,1)],axis=1)

train_y = np.transpose(deltthreshold[bigfiveparameter])

(train_x, test_x ,train_y, test_y) = train_test_split(merge_train_x.T, train_y, test_size = 0.3)
random_y = np.random.randint(2,size = 24)
predict_0 = np.zeros(24)



clf = RandomForestClassifier(max_depth=30, n_estimators=300, random_state=42)
clf.fit(train_x, train_y)
#
#Feature Importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")

with open('word_'+str(inputsize)+'.txt',mode='r') as f:
	word_100 = f.read()
word_100 = eval(word_100)

#for f in range(100):
#    print("%d. feature %d  %s (%f)" % (f + 1,indices[f],word_100[indices[f]], importances[indices[f]]))




# Plot the impurity-based feature importances of the forest

plt.figure()
plt.title("Feature importances")
plt.bar(range(train_x.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), indices)
plt.xlim([-1, train_x.shape[1]])
plt.savefig('fig1.png')



passto = []

passto.append(list(importances))
y_pred = clf.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)
#print('Accuracy: {}'.format(accuracy))
passto.append(accuracy)
accuracy = accuracy_score(test_y, random_y)
#print('Accuracy: {}'.format(accuracy))
passto.append(accuracy)
accuracy = accuracy_score(test_y, predict_0)
#print('Accuracy: {}'.format(accuracy))
passto.append(accuracy)
print(passto,file=sys.stderr)

