class TrainSCGParams(object):
	 def __init__(self,epochs = 100,min_grad = 1e-06,max_fail=6,goal=0,init_lambda=5e-07,sigma=5e-05):
	 	self.epochs = epochs
	 	self.min_grad = min_grad
	 	self.max_fail = max_fail
	 	self.goal = goal
	 	self.init_lambda = init_lambda
	 	self.sigma = sigma
