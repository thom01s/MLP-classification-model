import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import keras.datasets

data = keras.datasets.mnist.load_data(path="mnist.npz")

X_train = data[0][0]
y_train = data[0][1]
X_test = data[1][0]
y_test = data[1][1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = X_train.shape[1]
input_size = image_size * image_size

X_train = np.reshape(X_train, [-1, input_size])
X_train = X_train.astype('float32') / 255
X_test = np.reshape(X_test, [-1, input_size])
X_test = X_test.astype('float32') / 255

model = Sequential()
model.add(Dense(64, activation='tanh', input_dim=input_size))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["binary_crossentropy", "accuracy"])

history = model.fit(X_train, y_train, epochs=50, verbose=0)

#%%
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#%%
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

#%%
score = model.evaluate(X_test, y_test)
print("\nAcurácia: ", score[-1])
print("\nPredições:")
pred = model.predict(X_test)
print(pred)