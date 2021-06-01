import cv2
from random import randrange
# import randrange

# Load some pre_trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose photo
# img = cv2.imread('face.jpg') # import from my folder
# webcam_img = cv2.imread('webcam.jpg') # import from my folder

# Realtime video
webcam = cv2.VideoCapture(0)

while True:

    # Read the frame
    succesful_frame_read, frame = webcam.read()

    # Convert the grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    
    # Draw the rectangles around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x + w,y + h) ,(0,255,0),2) # (image,top left  , bottom right,color(BGR()),thickness)
    
    cv2.imshow('Face detector',frame)
    key = cv2.waitKey(1) # 1 miliseconds

    # Stop if Q pressed
    if key == 81 or key == 113:
        break


# Release webcam
webcam.release()
print("Code completed")