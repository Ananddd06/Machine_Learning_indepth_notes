import numpy as np

class LinearRegression:
    def __init__(self):
        self.teta = None
    
    def fit(self , X , y):
        if y.ndim == 1:
            y = y.reshape(-1,1)
        
        X_b = np.hstack([np.ones((X.shape[0] , 1)) , X])

        self.teta = np.linalg.inv(X_b.T @ X_b).dot(X_b.T).dot(y)
    
    def predict(self ,X):
        if self.teta is None:
            raise Exception("Model is not trained yet")
        
         
        X_b = np.hstack([np.ones((X.shape[0] , 1)) , X])

        return X_b @ self.teta

    def get_params(self):

        return self.teta

# Sample training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get learned parameters
theta = model.get_params()
print("Theta (parameters):", theta.ravel())

# Make predictions
preds = model.predict(X)
print("Predictions:", preds.ravel())



