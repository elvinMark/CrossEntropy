import numpy as np 

def sigmoid(x,diff=False):
	if diff:
		return x*(1-x)
	return 1/(1 + np.exp(-x))

def relu(x,diff=False):
	out = x.copy()
	if diff:
		out[x<0] = 0.1
		out[x>=0] = 1
		return out
	out[x<0] = 0.1*x[x<0]
	return out

def linear(x,diff=False):
	if diff:
		return np.ones(x.shape)
	return x

def tanh(x,diff=False):
	if diff:
		return (1-x**2) / 2
	return (1-np.exp(-x))/(1 + np.exp(-x))

class ELayer:
	def __init__(self,nin,nout,act_fun="tanh"):
		self.nin = nin
		self.nout = nout
		self.act_fun = act_fun
		self.w = np.random.random([nin,nout])
		self.bias = np.random.random([1,nout])
	def forward(self,indata):
		self.i = indata
		self.ones = np.ones([len(indata),1])
		self.o = indata.dot(self.w) + self.ones.dot(self.bias)
		if self.act_fun == "sigmoid":
			self.o = sigmoid(self.o)
		elif self.act_fun == "tanh":
			self.o = tanh(self.o)
		elif self.act_fun == "relu":
			self.o = relu(self.o)
		elif self.act_fun == "linear":
			self.o = linear(self.o)
		else:
			self.o = sigmoid(self.o)	
		return self.o
	def backward(self,err):
		if self.act_fun == "sigmoid":
			self.delta = sigmoid(self.o,diff=True)*err
		elif self.act_fun == "tanh":
			self.delta = tanh(self.o,diff=True)*err
		elif self.act_fun == "relu":
			self.delta = relu(self.o,diff=True)*err
		elif self.act_fun == "linear":
			self.delta = linear(self.o,diff=True)*err
		else:
			self.delta = sigmoid(self.o,diff=True)*err
		return self.delta.dot(self.w.T)
	def update(self,alpha=1):
		self.w = self.w - alpha*self.i.T.dot(self.delta)
		self.bias = self.bias - alpha*self.ones.T.dot(self.delta)

class ESoftMax:
	def __init__(self):
		self.sm = None
	def forward(self,indata):
		self.sm = []
		for elem in indata:
			tmp = np.exp(elem)/sum(np.exp(elem))
			self.sm.append(tmp)
		self.sm = np.array(self.sm)
		return self.sm
	def backward(self,target):
		err = []
		for elem,out in zip(target,self.sm):
			out[elem] -= 1
			err.append(out)
		err = np.array(err)
		return err
	def update(self,alpha=1):
		pass
class ENeuralNetwork:
	def __init__(self):
		self.layers = []
		self.softmax = False
	def add_layer(self,nin,nout,act_fun="sigmoid"):
		self.layers.append(ELayer(nin,nout,act_fun=act_fun))
	def add_softmax(self):
		self.layers.append(ESoftMax())
		self.softmax = True
	def forward(self,indata):
		o = indata
		for l in self.layers:
			o = l.forward(o)
		return o
	def backward(self,err):
		o = err
		for l in reversed(self.layers):
			o = l.backward(o)
	def update(self,alpha=1):
		for l in self.layers:
			l.update(alpha=alpha)
	def train(self,indata,target,maxIt=100,alpha=1):
		for i in range(maxIt):
			o = self.forward(indata)
			if self.softmax:
				self.backward(target)
			else:
				err = o - target
				self.backward(err)
			self.update(alpha=alpha)

