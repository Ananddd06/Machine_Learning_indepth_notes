# 📦 Machine Learning Algorithms From Scratch (NumPy Only)

This document contains a **complete mini ML library** implemented from scratch using NumPy:

- Optimizers
- Metrics
- Linear Models
- Tree Models
- Clustering
- Dimensionality Reduction
- SVM
- Gradient Boosting
- Anomaly Detection
- Autoencoder
- Pipeline

---

# 🔧 Imports

```python
import numpy as np
```

---

# ⚙️ OPTIMIZERS

```python
class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, w, dw):
        return w - self.lr * dw


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, w, dw):
        return w - self.lr * dw


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0
    def update(self, w, dw):
        self.v = self.beta * self.v + (1 - self.beta) * dw
        return w - self.lr * self.v


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = 0
    def update(self, w, dw):
        self.s = self.beta * self.s + (1 - self.beta) * (dw**2)
        return w - self.lr * dw / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.m, self.v, self.t = 0, 0, 0
        self.beta1, self.beta2 = 0.9, 0.999
        self.eps = 1e-8

    def update(self, w, dw):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

# 📊 METRICS

```python
def mse(y, y_pred): return np.mean((y - y_pred)**2)
def rmse(y, y_pred): return np.sqrt(mse(y, y_pred))
def mae(y, y_pred): return np.mean(np.abs(y - y_pred))
def accuracy(y, y_pred): return np.mean(y == y_pred)

def precision(y, y_pred):
    tp = np.sum((y==1)&(y_pred==1))
    fp = np.sum((y==0)&(y_pred==1))
    return tp/(tp+fp+1e-8)

def recall(y, y_pred):
    tp = np.sum((y==1)&(y_pred==1))
    fn = np.sum((y==1)&(y_pred==0))
    return tp/(tp+fn+1e-8)

def f1(y, y_pred):
    p, r = precision(y,y_pred), recall(y,y_pred)
    return 2*p*r/(p+r+1e-8)

def confusion_matrix(y, y_pred):
    tp = np.sum((y==1)&(y_pred==1))
    tn = np.sum((y==0)&(y_pred==0))
    fp = np.sum((y==0)&(y_pred==1))
    fn = np.sum((y==1)&(y_pred==0))
    return np.array([[tn, fp],[fn, tp]])
```

---

# 📈 LINEAR REGRESSION

```python
class LinearRegression:
    def __init__(self, opt, epochs=1000):
        self.opt = opt
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            dw = X.T @ (y_pred - y) / len(y)
            db = np.mean(y_pred - y)
            self.w = self.opt.update(self.w, dw)
            self.b = self.opt.update(self.b, db)

    def predict(self, X):
        return X @ self.w + self.b
```

---

# 🔐 LOGISTIC REGRESSION

```python
class LogisticRegression:
    def __init__(self, opt, epochs=1000):
        self.opt = opt
        self.epochs = epochs

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            p = self.sigmoid(X @ self.w + self.b)
            dw = X.T @ (p - y) / len(y)
            db = np.mean(p - y)
            self.w = self.opt.update(self.w, dw)
            self.b = self.opt.update(self.b, db)

    def predict(self, X):
        return (self.sigmoid(X @ self.w + self.b) > 0.5).astype(int)
```

---

# 👥 KNN

```python
class KNN:
    def __init__(self, k=3): self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y

    def predict(self, X):
        preds = []
        for x in X:
            d = np.linalg.norm(self.X - x, axis=1)
            idx = np.argsort(d)[:self.k]
            preds.append(np.bincount(self.y[idx]).argmax())
        return np.array(preds)
```

---

# 🔵 KMEANS

```python
class KMeans:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X):
        c = X[np.random.choice(len(X), self.k)]
        for _ in range(100):
            clusters = [[] for _ in range(self.k)]
            for i,x in enumerate(X):
                idx = np.argmin(np.linalg.norm(x-c,axis=1))
                clusters[idx].append(i)
            new_c = np.array([X[c].mean(0) for c in clusters])
            if np.allclose(c,new_c): break
            c = new_c
        self.centroids = c
```

---

# 📉 PCA

```python
class PCA:
    def __init__(self, n):
        self.n = n

    def fit(self, X):
        X = X - X.mean(0)
        cov = np.cov(X,rowvar=False)
        eigval,eigvec = np.linalg.eig(cov)
        idx = np.argsort(eigval)[::-1]
        self.comp = eigvec[:,idx[:self.n]]

    def transform(self,X):
        return (X-X.mean(0)) @ self.comp
```

---

# 🌳 DECISION TREE

```python
class DecisionTree:
    def __init__(self, depth=10):
        self.depth = depth

    def gini(self,y):
        return 1 - sum((np.sum(y==c)/len(y))**2 for c in np.unique(y))

    def split(self,X,y):
        best=None
        for f in range(X.shape[1]):
            for t in np.unique(X[:,f]):
                l = y[X[:,f]<=t]
                r = y[X[:,f]>t]
                if len(l)==0 or len(r)==0: continue
                g = self.gini(y) - (len(l)/len(y))*self.gini(l)-(len(r)/len(y))*self.gini(r)
                if best is None or g>best[0]:
                    best=(g,f,t)
        return best[1:] if best else (None,None)

    def build(self,X,y,d=0):
        if len(set(y))==1 or d==self.depth:
            return np.bincount(y).argmax()
        f,t = self.split(X,y)
        if f is None:
            return np.bincount(y).argmax()
        left = self.build(X[X[:,f]<=t],y[X[:,f]<=t],d+1)
        right = self.build(X[X[:,f]>t],y[X[:,f]>t],d+1)
        return (f,t,left,right)

    def fit(self,X,y):
        self.tree = self.build(X,y)

    def predict_one(self,x,node):
        if not isinstance(node,tuple): return node
        f,t,l,r = node
        return self.predict_one(x,l if x[f]<=t else r)

    def predict(self,X):
        return np.array([self.predict_one(x,self.tree) for x in X])
```

---

# 🌲 RANDOM FOREST

```python
class RandomForest:
    def __init__(self,n=10):
        self.n=n

    def fit(self,X,y):
        self.trees=[]
        for _ in range(self.n):
            idx = np.random.choice(len(X),len(X),True)
            t=DecisionTree()
            t.fit(X[idx],y[idx])
            self.trees.append(t)

    def predict(self,X):
        preds=np.array([t.predict(X) for t in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(),0,preds)
```

---

# ⚔️ SVM

```python
class SVM:
    def __init__(self, lr=0.001, lam=0.01, epochs=1000):
        self.lr, self.lam, self.epochs = lr, lam, epochs

    def fit(self, X, y):
        y = np.where(y==0,-1,1)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i,x in enumerate(X):
                cond = y[i]*(x@self.w+self.b)>=1
                if cond:
                    dw = 2*self.lam*self.w
                else:
                    dw = 2*self.lam*self.w - y[i]*x
                    self.b -= self.lr*(-y[i])
                self.w -= self.lr*dw

    def predict(self,X):
        return np.sign(X@self.w+self.b)
```

---

# 🚀 GRADIENT BOOSTING

```python
class GradientBoosting:
    def __init__(self,n=50,lr=0.1):
        self.n,self.lr=n,lr

    def fit(self,X,y):
        self.models=[]
        self.base=y.mean()
        pred=np.full_like(y,self.base)

        for _ in range(self.n):
            res=y-pred
            t=DecisionTree()
            t.fit(X,res)
            pred+=self.lr*t.predict(X)
            self.models.append(t)

    def predict(self,X):
        pred=np.full(X.shape[0],self.base)
        for m in self.models:
            pred+=self.lr*m.predict(X)
        return pred
```

---

# 🌲 ISOLATION FOREST (Simplified)

```python
class IsolationForest:
    def fit(self,X):
        self.mean=X.mean(0)
        self.std=X.std(0)

    def predict(self,X,th=3):
        z=np.abs((X-self.mean)/(self.std+1e-8))
        return (z>th).any(1).astype(int)
```

---

# 🧠 AUTOENCODER

```python
class Autoencoder:
    def __init__(self,d,h):
        self.W1=np.random.randn(d,h)*0.01
        self.W2=np.random.randn(h,d)*0.01

    def relu(self,x): return np.maximum(0,x)

    def fit(self,X,lr=0.01,epochs=1000):
        for _ in range(epochs):
            h=self.relu(X@self.W1)
            out=h@self.W2
            d=out-X
            self.W2-=lr*(h.T@d)
            self.W1-=lr*(X.T@(d@self.W2.T*(h>0)))

    def anomaly(self,X):
        h=self.relu(X@self.W1)
        out=h@self.W2
        return ((X-out)**2).mean(1)
```

---

# 🔗 PIPELINE

```python
class Pipeline:
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return self.metric(y, preds)
```

---
