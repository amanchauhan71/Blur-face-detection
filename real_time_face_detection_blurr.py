# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:22:03 2022

@author: Aman Chauhan
"""

import face_recognition
import cv2

#capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

#initializethe the array variable to hold all face locations in the frame
all_face_locations = []

#Create an outer while Loop to loop through each frame of video
#loop through every frame in video
while True:
    
    #Get single frame of video as images
    #get current frame
    ret , current_frame = webcam_video_stream.read()
    
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    
    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample,model
    all_face_location = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog') 
    
    #looping through the face locations
    for index, current_face_location in enumerate(all_face_location):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        # Change the magnitude to fit the actual size video frame
        top_pos     = top_pos*4
        right_pos   = right_pos*4
        bottom_pos  = bottom_pos*4
        left_pos    = left_pos*4
        
        #printing the location of current face
        print('Found face {} at top:{},right:{},bottom:{},Left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        
        #slicing the current face from main image
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        #blur the sliced face and save it to the same array itself
        current_face_image= cv2.GaussianBlur(current_face_image, (99,99), 30)
        #paste the blurr face into the actual face
        current_frame[top_pos:bottom_pos,left_pos:right_pos]=current_face_image
        
        #draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        
    #showing the current face with rectangle drawn
    cv2.imshow("Webcame video",current_frame)
        
        #break the infinite loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

#release the stream and cam
#close all open cv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()
        
        