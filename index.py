import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(x_test.shape)
#change shape to 3D
x_train=x_train.reshape(60000,28,28,1).astype("float32")
x_test=x_test.reshape(10000,28,28,1).astype("float32")

#preprocess data
x_train=x_train/255.0
x_test=x_test/225.0

image1=x_train[1]
print(image1)
#image1 is an array
print(image1.max())
print(image1.min())
plt.imshow(image1)
plt.title("Second image")
plt.show()

#converts number to array
from tensorflow.keras.utils import to_categorical
y_cat_train=to_categorical(y_train)
y_cat_test=to_categorical(y_test)
print("Object 2=", y_cat_train[1])
print(y_cat_train.shape)

#model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(4,4), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
print(model.summary())

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss', patience=2)
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_cat_train, epochs=10, batch_size=64, callbacks=[early_stop, checkpoint], validation_split=0.1,)
from tensorflow.keras.models import load_model
best_model = load_model('best_model.h5')
pred=best_model.predict(x_test)
test_loss, test_accuracy= best_model.evaluate(x_test, y_cat_test)
y_pred = np.argmax(pred, axis=1)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True,fmt='d', cmap="coolwarm")
plt.xlabel("Predicted label")
plt.ylabel("true label")
plt.title("Confusion matrix")
plt.show()
