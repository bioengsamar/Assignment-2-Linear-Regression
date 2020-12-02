import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        
    #cost function
    def computeCost(self,x, y, theta):
        j=(1/(2*len(x)))*np.sum(np.square((x * theta.T)-y))
        return j
    
    # gradient Descent
    def gradientDescent(self, x, y):
        global theta_new
        theta_new=np.matrix(np.zeros(x.shape[1]))
        cost=np.zeros(self.n_iters)
        
        for i in range(self.n_iters):
            h=(x * theta_new.T) - y
            for j in range(theta_new.shape[1]):
                #update theta_new (weights)
                theta_new[0,j] -=((self.lr/len(x)) * np.sum(np.multiply(h,x[:,j])))
                
            cost[i]=self.computeCost(x, y, theta_new)
            
        return theta_new, cost

    # train model
    def fit(self, x, y):
        #convert to matrices
        x=np.matrix(x)
        y=np.matrix(y)
        
        # gradient descent
        self.gradientDescent(x, y)
        
     
    # test model
    def predict(self, x):
        x=np.matrix(x)
        y_predicted=x * theta_new.T #prediction
        return y_predicted
    
    #Evaluate Performance>> RMSE
    def EvaluatePerformance(self, actual, predicted):
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        RMSE=np.sqrt(mean_error)
        return RMSE