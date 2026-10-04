import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from keras.models import load_model 

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def cnn_algo(name):
    # Recreate the exact same model, including its weights and the optimizer
    classes =['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 
        'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 
        'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 
        'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 
        'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 
        'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 
        'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 
        'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 
        'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 
        'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 
        'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 
        'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 
        'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 
        'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 
        'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 
        'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']
    da = name
    print(da)
    directory = os.getcwd()
    print(directory)

    loaded_model = load_model("Model/ayur_model_5.h5")
    path = name
    print(path)
    img = tf.keras.preprocessing.image.load_img(da, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = loaded_model.predict(img_array)
    # print(predictions[0]*100, "\n", classes)
    print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")
    return classes[np.argmax(predictions)],predictions[0][np.argmax(predictions)]*100
   
# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# loop over the frames from the video stream


while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    #frame = imutils.resize(frame, width=1000)
    cv2.imwrite("frame_test.jpg", frame )
    image_data = 'frame_test.jpg' 
    
    # Reading an image in grayscale mode
    image = cv2.imread(image_data, 0)

    # Window name in which image is displayed
    window_name = 'image'

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, image)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()           
    cnn_algo(image_data)
    time.sleep(10)