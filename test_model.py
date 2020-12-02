from sklearn.model_selection import train_test_split
import pandas as pd
from linear_regression import LinearRegression


def read_Data(path):
    data = pd.read_csv(path,header=None)
    data = (data - data.mean()) / data.std() # rescaling data
    data.insert(0,'X0',1)  # add ones column
    array=data.values
    
    if path == "univariateData.dat":
        x=array[:,0:2] #features 
        y=array[:,2:3] #target 
    else:
        x=array[:,0:3] #features 
        y=array[:,3:4] #target
        
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
    
    #train & test data
    regressor = LinearRegression(learning_rate=0.1, n_iters=100)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    RMSE=regressor.EvaluatePerformance(y_test,predictions) 
    return RMSE


if __name__ == "__main__":
    path1="univariateData.dat"
    path2="multivariateData.dat"
    
    print('RMSE_univariateData=',read_Data(path1)) #[0.47056066]
    print('RMSE_multivariateData=',read_Data(path2)) #[0.59231199]
    