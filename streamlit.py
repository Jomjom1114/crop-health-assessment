import streamlit as st
from keras.models import load_model
from skimage import transform
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = load_model('model3.h5')
model2 = load_model('modellast.h5')


def pred(image):
   st.image(image)
   shape=((128,128,3))
   test_image = image.resize((128,128))
   test_image= tf.keras.preprocessing.image.img_to_array(test_image)
   test_image= test_image/255.0
   test_image=np.expand_dims(test_image, axis=0)
   predictions=model.predict(test_image)
   #st.write(predictions[0])
   return predictions[0]

def pred2(test_image):
   st.image(test_image)
   #shape=((256,256,3))
   #test_image = image.resize((256,256))
   test_image= tf.keras.preprocessing.image.img_to_array(test_image)
   test_image= test_image/255.0
   test_image=np.expand_dims(test_image, axis=0)
   predictions=model2.predict(test_image)
   #st.write(predictions[0])
   class_names=['Tomato__Bacterial_spot', 'The Leaf Has Tomato Early Disease', 'The Leaf Has Tomato Late Blight Disease', 'Tomato_Leaf_Mold', 
               'Tomato_Septoria_leaf_spot','Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
               'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'The Leaf Is Healthy']

   return class_names[np.argmax(predictions)]

def main():
    st.title('Crop Health Assessment Web App')

    file = st.file_uploader("Please Upload A Tomato Leaf Image", type=["jpg", "png","jpeg"])
   
    if file is None:
     st.text("Waiting...")
   
    else:
     image = Image.open(file)
     if st.button('Binary Classification'):
       result = pred(image)
       if (result > 0.5):
         st.write('The Leaf Is Diseased')
       else:
         st.write('The Leaf Is Healthy')
         
     if st.button('Multiclass Classification'):
       result = pred2(image)

       st.write(result)

if __name__ == '__main__':
    main()
