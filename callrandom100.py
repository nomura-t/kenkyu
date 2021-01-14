import subprocess
import numpy as np
import matplotlib.pyplot as plt
from empath import Empath
import json

importances_list = []
accuracy_list = []
inputsize = 300
for i in range(100):
	proc = subprocess.run(["python3","randamforest.py"],stderr=subprocess.PIPE)
	
	passto = eval(proc.stderr)
	
	importances_list.append(passto[0])
	accuracy_list.append(passto[1:])

	
#Feature Importance
importances =np.mean(np.array(importances_list),axis=0)
accuracy = np.mean(np.array(accuracy_list),axis=0)
std = np.std(np.array(importances_list), axis=0)
std_accuracy = np.std(np.array(accuracy_list), axis=0)

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

with open('word_'+str(inputsize)+'.txt',mode='r') as f:
	word_100 = f.read()
word_100 = eval(word_100)


for f in range(100):
    print("%d. feature %d  %s (%f)" % (f + 1,indices[f],word_100[indices[f]], importances[indices[f]]))

print('proposed:\t{:.4} ({:.4})'.format(accuracy[0],std_accuracy[0]))
print('random:\t\t{:.4} ({:.4})'.format(accuracy[1],std_accuracy[1]))
print('predict0:\t{:.4} ({:.4})'.format(accuracy[2],std_accuracy[2]))

# Plot the impurity-based feature importances of the forest

plt.figure()
plt.title("Feature importances")
plt.bar(range(100), importances[indices[:100]],
        color="r", yerr=std[indices[:100]], align="center")
plt.xticks(range(100), indices[:100])
plt.xlim([-1, 100])
plt.savefig('importances_mean_'+str(inputsize)+'.png')

#empath
lexicon = Empath()
analyzevalue_dict = {}
analyzevalue_list = []
empathcount = {}
for i in range(len(word_100)):
	empathapply =lexicon.analyze(word_100[i].lower())
	for k in empathapply.keys():
		analyzevalue_dict[k]=analyzevalue_dict.get(k,0)+empathapply[k]*importances[i]
		empathcount[k]=empathcount.get(k,0)+empathapply[k]
copyanalyzevalue_dict = analyzevalue_dict.copy()
for k in copyanalyzevalue_dict:
	if analyzevalue_dict[k] == 0:
		analyzevalue_dict.pop(k)
		empathcount.pop(k)
d =  [(v,k) for k, v in analyzevalue_dict.items()]
d.sort()
d.reverse()
print(d)
e =  [(v,k) for k, v in empathcount.items()]
e.sort()
e.reverse()
with open('empathlist.txt',mode='a') as f:
	f.write(json.dumps(analyzevalue_dict))
