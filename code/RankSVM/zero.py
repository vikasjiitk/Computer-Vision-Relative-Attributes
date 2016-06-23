#This code is for zero shot learning
import csv
import numpy as np
from sklearn import mixture
import scipy
from scipy.stats import multivariate_normal
from sklearn import metrics
from copy import deepcopy
f = open('rankcombinedunseen.txt', 'r')  #this should contain the RankSpace vector for each training and testing image
b=f.read()
a=eval(b)
trueRank = np.asarray(a)
print (len(trueRank))
f = open('nlabel.txt', 'r') #labels for the images
labels = [eval(line.strip()) for line in f]

n_att = len(trueRank[0])
n_class= 8 #number of seen classes
n = len(labels)
### Rank to be loaded from rank.txt
Rank = np.zeros((n,n_att))
data= [[] for i in range(n_class)]
train_len=680
for i in range(train_len):
	data[labels[i]-1].append(trueRank[i])
for i in range(n_class):
	print(len(data[i-1]))
gaudist=[]
mean=[]
covar=[]

unseen=[[[2,3],[1,3],[3,7],[5,7],[2,3],[4,8],[4,3],[2,1],[1,6],[8,5]] # new
,[[2,3],[1,3],[7,1],[1,3],[2,3],[4,8],[2,8],[5,2],[5,1],[8,5]]
,[[5,6],[7,6],[8,4],[6,8],[4,5],[6,7],[4,3],[7,3],[1,6],[2,6]]];

for i in range(n_class):
	g=mixture.GMM(n_components=1,covariance_type='full')
	g.fit(data[i])
	mean.append(g.means_[0])
	covar.append(g.covars_[0])
	gaudist.append(g)
for i in range(len(unseen)):
	me=[]
	s=[0 for j in range(n_class)]
	support=0
	for j in range(len(unseen[i])):
		me.append((mean[unseen[i][j][0]-1][j] + mean[unseen[i][j][1]-1][j])/2)
		s[unseen[i][j][0]-1]=1
		s[unseen[i][j][1]-1]=1
	fl=0
	co=[]
	for j in range(n_class):
		if fl:
			support+=1
			co+=s[j]*covar[j]
		elif s[j]:
			support+=1
			co=deepcopy(covar[j])
			fl=1
	mean.append(np.array(me))
	covar.append(np.array(co)/support)
predclass=[]
for i in range(len(labels[train_len:])):
	p=0
	clas=1
	for j in range(len(mean)):
		# print (j)
		temp=multivariate_normal(mean[j],covar[j]).pdf(trueRank[train_len+i])
		# print(temp)
		if(temp> p):
			p=temp
			clas=j+1
	predclass.append(clas)
	print (labels[train_len+i],clas)
print(metrics.classification_report(predclass,labels[train_len:]))
