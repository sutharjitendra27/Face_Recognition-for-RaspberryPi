import face_recognition
import cv2
import numpy as np
import os 
from datetime import datetime
from weakref import Ref
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db 
from datetime import datetime
import re

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# credentials of dtaabase
cred = credentials.Certificate(
    'sih-demo-project-firebase-adminsdk-9cqbs-6c654809a2.json')

firebase_admin.initialize_app(cred, {

    'databaseURL': 'https://sih-demo-project-default-rtdb.firebaseio.com/'

})

ref = db.reference('Activity/')

# getting the 
path = 'images'
images = []
personName = []
myList = os.listdir(path)
#printing the names
print(myList)

# split the name and extensions
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)

#creating face encodings using a function
def faceEncodings(images):
    encodeList = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

known_face_encodings = faceEncodings(images)
print("All encodings created!")

# hog algorithm is used to do this encoding

time_now = datetime.now()
tStr = time_now.strftime('%H:%M')
dStr = time_now.strftime('%d/%m/%Y')


# creating Function for marking attendance
def attendance(name):

        nameList = []
        
        #creating a new variable and replacing space from name variable (between name and contact)
        new = name.replace(' ', '')
        
        #splitting string and numbers
        temp = re.compile("([a-zA-Z]+)([0-9]+)")
        
        #storing the string and contact number into tuple
        res = temp.match(new).groups()
        
        #string name
        str_Name = res[0]
        #stroring contact number
        str_Contact = res[1]
        
        for name in nameList:
            break
                                               
        if name not in nameList :
            nameList.append(name)

            data = {'Name': str_Name,
                    'Contact': str_Contact,
                    'Time': tStr,
                    'Date': dStr,
                    'status': 'In',
                    }
            
            print(data)
            ref.push(data)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = personName[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(10) == 13:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()