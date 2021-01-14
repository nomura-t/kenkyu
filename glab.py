import pickle
import sys, codecs

#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

with open('glvvec.dict.pkl','rb') as f:
	glv = pickle.load(f)
wordlist = []
for i in range(80):
	with open("./data/file"+str(i*3+1)+'.txt','rb') as f:
		for line in f:
			for Word in line.split():
				try:
					#wordlist.append(glv[Word])
					wordlist.append(glv[Word.decode('utf-8')])
				except KeyError:
					pass

	with open('./vector/vector'+str(i)+'.txt','wb') as w:
		pickle.dump(wordlist,w)
	wordlist = []

