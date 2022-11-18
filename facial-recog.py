import cv2
import numpy as np
import face_recognition
import os


path="faceImages"
images=[]
classNames=[]
myList = os.listdir(path)


for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # get name only




def findEncodings(images):
    encodeList =[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



encodeListKnown= findEncodings(images)
print("encoding complete!")

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesInCurrentFrame = face_recognition.face_locations(imgS) #multiple faces
    encodingsCurrentFrame=face_recognition.face_encodings(imgS,facesInCurrentFrame)

    for encodeFace,faceLocation in zip(encodingsCurrentFrame,facesInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLocation
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow("Webcam",img)
    cv2.waitKey(1)









# faceLocation = face_recognition.face_locations(imgWill)[0]
# encodeWill=face_recognition.face_encodings(imgWill)[0]
# cv2.rectangle(imgWill,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(0,255,0),2)


# faceLocationTest = face_recognition.face_locations(imgWillTest)[0]
# encodeWillTest=face_recognition.face_encodings(imgWillTest)[0]
# cv2.rectangle(imgWillTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(0,255,0),2)


# results= face_recognition.compare_faces([encodeWill],encodeWillTest) # see if its the same person

# faceDist= face_recognition.face_distance([encodeWill],encodeWillTest)
# print(results,faceDist)

# cv2.putText(imgWillTest,f'{results} {round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

# cv2.imshow('Will Smith',imgWill)
# cv2.imshow('Will Smith Test',imgWillTest)

# cv2.waitKey(0)









