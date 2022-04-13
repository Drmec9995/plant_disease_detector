import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as sl
import pandas as pd



# loading model
model = load_model(f'C:/Users/ABDULBASIT/Desktop/plant_disease_detector/plant_disease_detect.h5')

# class name
class_name =['Corn-Common_rust','Potato-Early_blight','Tomato-Bacterial_spot']

sl.title('Plant disease detection')
sl.markdown('Upload an image of a plant')

# uploading the leaf image...
plant_image = sl.file_uploader('choose an image...',type='jpg')
submit = sl.button('predict')

# on predict button click

if submit:
    if plant_image is not None:
        # convert the file to an opencv image
        
        file_bytes = np.asarray(bytearray(plant_image.read()),dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes,1) 
        

        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image,(256,256))
        
        # convert image to 4 dimensional image
        opencv_image.shape = (1,256,256,3)
         
         # making prediction...
        mid_val = 35
        y_pred = model.predict(opencv_image)
        score = tf.nn.softmax(y_pred)
        score = score.numpy()* 100.0
        chart_data = pd.DataFrame(
        score,
        columns=class_name)
        sl.bar_chart(chart_data)
        if np.any(max(score) > mid_val):
            
            # displaying image
            sl.image(opencv_image,channels='RGB')
            sl.write(opencv_image.shape)
           
                
            result = class_name[np.argmax(y_pred)]
            
            
            sl.title(str('This looks more of a ' + result.split('-')[0] + ' leaf with ' + result.split('-')[1]))
    else:
        sl.write('''##Pls kind upload an image''')
        
         
        