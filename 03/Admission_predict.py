import pandas as pd

# Data Loading
path = './'
data = pd.read_csv(path + 'Admission.csv')

# Simple EDA
data.shape
data.head()

# Separation of features and target
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
print(x)
print(y)

# Separation of train set and test set
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=108)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# model define
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(4, activation='relu', input_dim=4))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))  # as this is Regression problem
model.summary() # show model architecture

# hyperparameter setting
model.compile(loss='mean_squared_error', optimizer='Adam')  # as this is Regression problem

# train
hist = model.fit(train_x, train_y, epochs=100, validation_split=0.2) # 1 epoch = 1 forward + 1 backpropagation

# plotting training and validation loss
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()
# inference
import numpy as np

y_pred = model.predict(np.array([[320, 120, 9.5, 4]]))
print(y_pred)
