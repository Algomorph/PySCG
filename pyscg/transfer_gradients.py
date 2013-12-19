import numpy as np

#output = tanh(a)
#da/dnet = 1-tanh^2(a)
def tanh_grad(output):
	return 1 - output**2

#da/dnet = 1
def lin_grad(output):
	return 1

grad_dict = {
	None: lin_grad,
	np.tanh: tanh_grad
}
