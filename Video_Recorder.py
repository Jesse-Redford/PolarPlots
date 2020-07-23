# -*- coding: utf-8 -*-
import sys
sys.path.append(r'c:\programdata\anaconda3\lib\site-packages')
import cv2
import time


video_capture = cv2.VideoCapture(0)
out = cv2.VideoWriter('comp_outpy.mp4v',cv2.VideoWriter_fourcc('I','4','2','0'),10, (640,480),0)

i=0
video_length = 20
start = time.time()
while True:
    now = time.time()
    
    if now-start >= video_length-1:
        break
    
    
    # read in frame from camera
    ret, frame = video_capture.read()
    
    out.write(frame) # 5620kb
    cv2.imshow('Local SVD Compression', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
      
video_capture.release()
out.release()
    
    


