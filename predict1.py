import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
import pyautogui
import math
import copy
import time

# parameters
x=0.5  
y=0.8  
threshold = 60  
bgSubThreshold = 50
learningRate = 0


# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")


# variables
isBgCaptured = 0   
triggerSwitch = False  

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)

categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE',6: '.'}


while camera.isOpened():
    ret, frame = camera.read()
    threshold = 60
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)  
    cv2.rectangle(frame, (int(x * frame.shape[1]), 0),
                 (frame.shape[1], int(y * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    

    if isBgCaptured == 1:
        img = removeBG(frame)
        img = img[0:int(y * frame.shape[0]),
                    int(x * frame.shape[1]):frame.shape[1]]
        img=cv2.resize(img,(64,64))
        cv2.imshow('mask', img)
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(rgb,(5,5),0)
        cv2.imshow('blur', blur)
        
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


            # Batch of 1
        result = loaded_model.predict(thresh.reshape(1, 64, 64, 1))
        prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  '.' : result[0][6]}
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
        # Displaying the predictions
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
        cv2.imshow("Frame", frame)
       
  
    k = cv2.waitKey(10)
    if k == 27:
        break
    elif k == ord('b'):  # bg start
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # reset
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')

camera.release()
cv2.destroyAllWindows()
