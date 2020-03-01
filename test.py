import neural as nn
import numpy as np 

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
ytmp = np.array([[0],[1],[1],[0]])

mlp = nn.ENeuralNetwork()
mlp.add_layer(2,5,act_fun="relu")
mlp.add_layer(5,3,act_fun="relu")
mlp.add_layer(3,2,act_fun="relu")
mlp.add_softmax()

mlp.train(x,y,alpha=0.1,maxIt=500)
print(mlp.forward(x))