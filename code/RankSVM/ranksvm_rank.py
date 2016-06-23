#This file can be used to compute O_ and S_ that are needed for SVM
cl=[1,86,171,256,341,426,511,596]
n=680
class_strength_main = [[6, 8, 7, 5, 2, 1, 4, 3]]
 # These list correpsonds to different attributes ranking between classes
 #[1, 2, 3, 5, 7, 6, 8, 4]
 # [5, 3, 2, 4, 8, 6, 1, 7]
 # ,[8, 4, 3, 2, 6, 7, 1, 5]
 # ,[5, 6, 7, 1, 3, 4, 8, 2]
 # ,[6, 7, 5, 8, 1, 2, 4, 3]
 # [4, 6, 5, 2, 1, 3, 7, 8],
 # ,[1, 2, 8, 7, 4, 6, 5, 3]
 # ,[7, 5, 1, 2, 6, 8, 3, 4]
 # [6, 4, 1, 3, 8, 7, 2, 5]

O_=[]
for i in range(len(class_strength_main)):
	y=[ [0 for k in range(n)] for j in range(len(cl)-1)]
	for j in range(len(cl)-1):
		y[j][cl[class_strength_main[i][j]-1]]=-1
		y[j][cl[class_strength_main[i][j+1]-1]]=1
	O_.append(y)
S_=[]
for i in range(len(class_strength_main)):
	y=[]
	for j in range(len(cl)):
		if (j==len(cl)-1):
			for k in range(cl[j]-1,n-1):
				x=[0 for b in range(n)]
				x[k]=-1
				x[k+1]=1
				y.append(x)
		else:
			for k in range(cl[j]-1,cl[j+1]-2):
				x=[0 for b in range(n)]
				x[k]=-1
				x[k+1]=1
				y.append(x)
	S_.append(y)
O_=O_[0]
for i in range(len(O_)):
	for j in range(len(O_[i])):
		print O_[i][j],
		if(j!=len(O_[i])-1):
			print ', ',
	print '\n'
S_=S_[0]
for i in range(len(S_)):
	for j in range(len(S_[i])):
		print S_[i][j],
		if(j!=len(S_[i])-1):
			print ', ',
	print '\n'
print len(O_)
print len(S_)
