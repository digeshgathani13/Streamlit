
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache
def load_image(img):
    im = Image.open(img)
    return im


face_cascade = cv2.CascadeClassifier("C://Digesh Personal//SimpliLearn//Capstone_Project//Streamlit//frontalFace10//haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C://Digesh Personal//SimpliLearn//Capstone_Project//Streamlit//frontalFace10//haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("C://Digesh Personal//SimpliLearn//Capstone_Project//Streamlit//frontalFace10//haarcascade_smile.xml")

def detect_faces(our_image):
    new_image = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_image, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #face detect
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # draw rectangle
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0), 4)       
    return img, faces
    

def detect_eyes(our_image):
    new_image = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_image, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh),(0, 255,0), 4)
    return img, eyes

        
def detect_smiles(our_image):
    new_image = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_image, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)    
    for(sx,sy,sw,sh) in smiles:
        cv2.rectangle(img, (sx,sy), (sx+sw,sy+sh),(0, 0, 255), 2)
    return img, smiles


def cartonize_image(our_image):
    new_image = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_image, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2)
    color=cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
    
    
def cannize_image(our_image):
    new_image = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_image, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def main():
    st.title("Face Detection App")
    st.text("Build with Streamlit and Opencv")
    
    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("select activity", activities)
    
    if choice == "Detection":
        st.subheader("Face Detection")
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            #st.write(type(our_image))
            st.image(our_image)
        
        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enhance_type == "Gray-Scale":
            new_image = np.array(our_image.convert("RGB"))
            img = cv2.cvtColor(new_image, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(gray)
            #st.write(new_image)
        elif enhance_type == "Contrast":
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)
        elif enhance_type == "Brightness":
            c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)
        elif enhance_type == "Blurring":
            new_image = np.array(our_image.convert("RGB"))
            img = cv2.cvtColor(new_image, 1)
            blur_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            st.image(blur_img)
        else:
            st.image(our_image, width=300)
        
        
        # face detection
        task = ["Faces", "Smiles", "Eyes", "Cannize", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Find Feature", task)
        if st.button("Process"):
            if feature_choice == "Faces":
                result_image, result_faces = detect_faces(our_image)
                st.image(result_image)
                st.success("Found {} faces".format(len(result_faces)))
            elif feature_choice == "Smiles":
                result_image, result_smiles = detect_smiles(our_image)
                st.image(result_image)
                st.success("Found {} smiles".format(len(result_smiles)))
            elif feature_choice == "Eyes":
                result_image, result_eyes = detect_eyes(our_image)
                st.image(result_image)
                st.success("Found {} eyes".format(len(result_eyes)))
            elif feature_choice == "Cartonize":
                result_image = cartonize_image(our_image)
                st.image(result_image)
            elif feature_choice == "Cannize":
                result_image = cannize_image(our_image)
                st.image(result_image)
    
    elif choice == "About":
        st.subheader("About")
        
        

if __name__ == "__main__":
    main()