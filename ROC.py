import pylab as pl
from math import log,exp,sqrt
result="/Users/pengwu5501/Desktop/pcky/sent1.txt"
db = [] #[score,nonclk,clk]
pos, neg = 0, 0
fs = open(result,'r').read().strip().split('\n')
for line in fs:
    nonclk,clk,score = line.split('\t')
    nonclk = int(nonclk)
    clk = int(clk)
    score = float(score)
    print([score,nonclk,clk])
    db.append([score,nonclk,clk])
    pos += clk
    neg += nonclk
    db = sorted(db, key=lambda x:x[0], reverse=True)
#calculate ROC position
#print(db)
xy_arr = []
tp, fp = 0., 0. 
for i in range(len(db)):
 	tp += db[i][2]
 	fp += db[i][1]
 	xy_arr.append([fp/neg,tp/pos])
auc = 0. 
prev_x = 0
for x,y in xy_arr:
 	if x != prev_x:
 		auc += (x - prev_x) * y
 		prev_x = x
print("the auc is %s."%auc)
x = [_v[0] for _v in xy_arr]
y = [_v[1] for _v in xy_arr]
pl.title("ROC curve of %s (AUC = %.4f)" % ('svm',auc))
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.plot(x, y)# use pylab to plot x and y
pl.show()# show the plot on the screen
