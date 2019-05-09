#import library
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)
#num of points
n_pts = 500
#labels, middle is 1 outside is 0, random state keeps same chart, noise is deviation,
#factor is the diameter of the inner circle vs outer
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

#creates a sequential neural net
model = Sequential()
#creates a dense hidden layer with 4 nodes and 2 inputs
model.add(Dense(4, input_shape=(2,), activation="sigmoid"))
#adds output layer,
model.add(Dense(1, activation="sigmoid"))
#metrics are very similar loss function, except they're not backpropegated, their used as
#ways to measure the algorithem
model.compile(Adam(lr=0.01), "binary_crossentropy", metrics=["accuracy"])
#batch size is how many times it backpropregates.
#epoch is # of times model runs through data
h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle="true")

#plots the accuracy of the model per epoch
plt.plot(h.history['acc'])
plt.legend(['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

#plots loss of model per epoch
plt.plot(h.history['loss'])
plt.legend(['loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.show()

#creates contours seperating the different data points
def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

#plots contours and datapoints. It also adds a new unlabeld point which the model predicts a value
plot_decision_boundary(X, y, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
x = 0
y = 0.75
point = np.array([[x, y]])
predict = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("Prediction is: ", predict)
plt.show()
