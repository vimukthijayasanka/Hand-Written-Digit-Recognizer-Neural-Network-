import numpy as np
import os
import cv2                        #For Load Images, Process Images
import matplotlib.pyplot as plt 
import tensorflow as tf           #For Machine Learning 

#Loading Dataset directly from tensorflow
mnist = tf.keras.datasets.mnist

#-------------Split Data-----------------------
# Training data - actually use to trained the model
# Testing data - use in order to assess the model
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#-------------Normalized data------------------- Scale 0 to 1
X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)

#----------Creating Neural Network Model------------
# model = tf.keras.models.Sequential()
# #Add layers into model
# model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #Flatten Layer - flatten a certain input shape (Basically think 25x25 px grid, its give 25x25 = 625 pixels flat line)
# model.add(tf.keras.layers.Dense(128,activation='relu')) #Dense Layer - Connect each neurons from each other layers neurons
# # 'relu' - rectify linear unit 
# model.add(tf.keras.layers.Dense(10,activation ='softmax')) #Output Layer - we have 0 to 9 digits its means we have 10 neurons
# # softmax is doing all the outputs turnup into one, gives the probability for each digit to be the right answer

# #-----------Compile the model--------------------
# model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# #----------Train the model----------------------
# model.fit(X_train,y_train,epochs=3)
# #epochs - basically how many iteraions going to see how many times is the model going to see the same data all over again

# model.save("handwritten_model.keras")
model = tf.keras.models.load_model('handwritten_model.keras')

#----------Testing accuracy of the model--------------
loss, accuracy, = model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

image_number = 0
while os.path.isfile(f"images/digit_{image_number}.jpg"):
    try:
        img = cv2.imread(f"images/digit_{image_number}.jpg")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number += 1



