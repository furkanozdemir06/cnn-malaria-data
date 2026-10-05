import streamlit as st 
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('malaria_data.TL.keras')
def process_image(img):
    img=img.resize((224,224))
    img=np.array(img)
    img=img/255.0 
    img=np.expand_dims(img,axis=0)
    return img
st.title('Malaria Prediction')
st.write('Upload Image, Predict Malaria!')
file=st.file_uploader('Select an Image',type=['jpg','jpeg','png'])
if file is not None: # Resim yuklenmisse burasi calisacak
    img=Image.open(file)
    st.image(img, caption='Uploaded Images')
    image=process_image(img)
    prediction=model.predict(image)[0][0]
    predicted_class=1 if prediction > 0.5 else 0
    class_names=['Malaria (Infected)', 'Not Malaria (Uninfected)']
    st.write(class_names[predicted_class])

