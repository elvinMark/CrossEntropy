import numpy as np 
import matplotlib.pyplot as plt 
import neural as nn
import cv2 as cv

ind = np.loadtxt("indatanew.dat",dtype=np.int16)
outd = np.loadtxt("outdatanew.dat",dtype=np.int8).tolist()

print(outd)

I = 255*np.ones([500,500,3])

mlp = nn.ENeuralNetwork()
mlp.add_layer(2,5,act_fun="relu")
mlp.add_layer(5,3,act_fun="relu")
mlp.add_layer(3,2,act_fun="relu")
mlp.add_softmax()
inputd = ind/500
mlp.train(inputd,outd,maxIt=1000,alpha=0.01)
print(mlp.forward(inputd))
print(outd)

for i in range(50):
	for j in range(50):
		test = np.array([[i/50,j/50]])
		o = mlp.forward(test)[0]
		if o[0]>o[1]:
			I = cv.rectangle(I,(10*i,10*j),(10*(i+1),10*(j+1)),(0,0,0.5),-1)
		else:
			I = cv.rectangle(I,(10*i,10*j),(10*(i+1),10*(j+1)),(0,0.5,0),-1)

for i,j in zip(ind,outd):
	if j == 0:
		I = cv.circle(I,(i[0],i[1]),10,(0,0,1),-1)
	else:
		I = cv.circle(I,(i[0],i[1]),10,(0,1,0),-1)

plt.imshow(I)
plt.show()