import streamlit as st
from keras.models import load_model
from skimage import transform
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = load_model('model3.h5')


def pred(image):
   st.image(image)
   shape=((128,128,3))
   test_image = image.resize((128,128))
   test_image= tf.keras.preprocessing.image.img_to_array(test_image)
   test_image= test_image/255.0
   test_image=np.expand_dims(test_image, axis=0)
   predictions=model.predict(test_image)
   st.write(predictions[0])
   return predictions[0]

def main():
    st.title('Crop Health Assessment Web App')

    file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
   
    if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    if st.button('Crop Health Assessment Result'):
       result = pred(image)

       #st.write(result)
       if (result > 0.5):
         st.write('Diseased')
       else:
         st.write('Healthy')

if __name__ == '__main__':
    main()
