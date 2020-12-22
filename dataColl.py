import cv2
import numpy as np
import copy
import math
import os

mode = 'train'
directory = 'data/'+mode+'/'

# parameters
cap_region_x_begin=0.5  
cap_region_y_end=0.8  
threshold = 60  
bgSubThreshold = 50
learningRate = 0
frame=0
thresh=0
isBgCaptured = 0   


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)


while camera.isOpened():
    ret, frame = camera.read()
    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             '.': len(os.listdir(directory+"/6"))}
    
    threshold = 60
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, ". : "+str(count['.']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)


    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    
  

    if isBgCaptured == 1:
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        img=cv2.resize(img,(64,64))
        cv2.imshow('mask', img)

            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(rgb,(5,5),0)
        cv2.imshow('blur', blur)
        
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


            
    k = cv2.waitKey(10)
    if k & 0xFF == 27:
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k & 0xFF == ord('0'):
            cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', thresh)
    elif k & 0xFF == ord('1'):
            cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', thresh)
    elif k & 0xFF == ord('2'):
            cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', thresh)
    elif k & 0xFF == ord('3'):
            cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', thresh)
    elif k & 0xFF == ord('4'):
            cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', thresh)
    elif k & 0xFF == ord('5'):
            cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', thresh)
    elif k & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['.'])+'.jpg', thresh)
            
    elif k == ord('r'):
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')

camera.release()
cv2.destroyAllWindows()
