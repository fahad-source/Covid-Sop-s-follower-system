#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import*
from tkinter import messagebox
import cv2
import time
import numpy as np
import mysql.connector
from plyer import notification
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
model = load_model('New folder (2)/model.h5') # cnn
 
# model accept below hight and width of the image
img_width, img_hight = 200, 200


# In[3]:



def fun():
        framewidth=640
        frameheight=480

        # Load the Cascade face Classifier
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # cap = cv2.VideoCapture('covid.mp4') # for video
        cap = cv2.VideoCapture(0) # for webc
        def empty(a):
            pass

        cv2.namedWindow("Controls")
        cv2.moveWindow("Controls",0,380)
        cv2.resizeWindow("Controls",600,180)
        cv2.createTrackbar("Scale","Controls",100,1000,empty)
        cv2.createTrackbar("Neig","Controls",5,50,empty)
        cv2.createTrackbar("Min area","Controls",0,100000,empty)
        cv2.createTrackbar("Brightness","Controls",0,255,empty)


        img_count_full = 0

        #parameters for text
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # org 
        org = (1, 1)
        class_lable=' '      
        # fontScale 
        fontScale = 1 #0.5
        # Blue color in BGR 
        color = (255, 0, 0) 
        # Line thickness of 2 px 
        thickness = 2 #1

        #sart reading images and prediction
        while True:
            #read image from webcam
            responce, color_img = cap.read()
            #if respoce False the break the loop
            if responce == False:
                break    


            camera_brightness=cv2.getTrackbarPos("Brightness","Controls")
            cap.set(10,camera_brightness)
            img2=cv2.resize(color_img,(framewidth,frameheight))
            gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            scale_value=1+(cv2.getTrackbarPos("Scale","Controls")/1000)
            neig=(cv2.getTrackbarPos("Neig","Controls")+1)
            img_count_full += 1

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray_img ,scale_value, neig) # 1.1, 3) for 1.mp4

            #take face then predict class mask or not mask then draw recrangle and text then display image
            img_count = 0
            for (x, y, w, h) in faces:
                area=w*h
                min_area=(cv2.getTrackbarPos("Min area","Controls"))
                org = (x-10,y-10)
                img_count +=1 
                color_face = color_img[y:y+h,x:x+w] # color face
                cv2.imwrite('pics/%d%dface.jpg'%(img_count_full,img_count),color_face)
                img = load_img('pics/%d%dface.jpg'%(img_count_full,img_count), target_size=(img_width,img_hight))
                img_count +=1 
                color_face = color_img[y:y+h,x:x+w] # color face
                cv2.imwrite('pics/%d%dface.jpg'%(img_count_full,img_count),color_face)
                img = load_img('pics/%d%dface.jpg'%(img_count_full,img_count), target_size=(img_width,img_hight))

                img = img_to_array(img)/255
                img = np.expand_dims(img,axis=0)
                pred_prob = model.predict(img)
                #print(pred_prob[0][0].round(2))
                pred=np.argmax(pred_prob)
                if pred==0:
                    class_lable = "Mask"
                    color = (0, 255, 0)
                    cv2.imwrite('mask/%d%dface.jpg'%(img_count_full,img_count),color_face)
                else:
                    start = time.time()
                    class_lable = "No Mask"
                    color = (0, 0, 255)
                    cv2.imwrite('without_mask/%d%dface.jpg'%(img_count_full,img_count),color_face)
                    end= time.time() 
                    total=(end-start)
                    print (total)
                    if total>=0.01:
                        notification.notify(title="Mask violation",message="someone is not wearing mask properly kindly check it",app_icon="mask.ico",timeout=1) 
                if area > min_area:
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(color_img, class_lable, org, font,  
                                           fontScale, color, thickness, cv2.LINE_AA) 

            cv2.imshow('LIVE face mask detection press q for close', color_img)
            cv2.moveWindow('LIVE face mask detection press q for close',0,0)
            cv2.resizeWindow("LIVE face mask detection press q for close",600,350)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
        fun()
        





# In[ ]:




