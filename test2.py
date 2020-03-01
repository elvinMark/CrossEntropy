import neural as nn
import numpy as np

mlp = nn.ENeuralNetwork();
mlp.add_layer(2,5,act_fun="relu")
mlp.add_layer(5,3,act_fun="relu")
mlp.add_layer(3,2,act_fun="relu")
mlp.add_softmax()
ind = np.array([[0,0],[0,1],[1,0],[1,1]])
outd = [0,1,1,0]

mlp.train(ind,outd,alpha=0.1,maxIt=1000)
print(mlp.forward(ind))
print(mlp.forward(ind))
print(mlp.forward(ind))