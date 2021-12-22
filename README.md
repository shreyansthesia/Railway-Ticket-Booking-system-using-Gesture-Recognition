
# Railway Ticket booking system using Gesture Recognition.    

A This project was developed in response to COVID-19 spread. When the Lockdown in India got uplifted, the Indian Railways were the prime hotspots and contributors to spreading the virus. Existing kiosks were operated by touch and thus nullified the meaning of Social Distancing. 

In view of this, I stated to work on the project to create a gesture based ticket booking system which can be operated through gestures and no physical contact is required to book the tickets.


## Dataset Used
The Dataset was created by me. It consist of 10000+ images with each gesture having its own folder. 
## Requirements

Install opencv.Note: pip install opencv-python does not have video capabilities. So I recommend to build it from source as described above.

Install TensorFlow:
```bash
  pip install tensorflow
```

Install tflearn:
```bash
  pip install tflearn
```

## Gesture Input
I am using OpenCV for capturing the user's hand gestures. The post processing on the captured images to highlight the contours and edges. like applying binary threshold, blurring and grey scale.

Here first, the image is converted to greyscale, then gaussian blur is applied with adaptive threshold.
## Screenshots
https://drive.google.com/drive/folders/1xmTIQuo1RNJFSLSHrBCMfooUbjyC12YI?usp=sharing

## Conclusion

So what next?
Well i thought why only gestures? I am planning to add voice support to book tickets.