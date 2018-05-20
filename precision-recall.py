import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
plt.figure(1) # create figure
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


x=[]
y=[]
f=open('sent1.txt','r')
lines=f.readlines()
for i in range(len(lines)/3):
    y.append(float(lines[3*i].strip().split(':')[1]))
    x.append(float(lines[3*i+1].strip().split(':')[1]))
f.close()
plt.figure(1)
plt.plot(x, y)
plt.show()
plt.savefig('p-r.png')