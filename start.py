import cv2
import numpy as np
import face_recognition


imgWill= face_recognition.load_image_file("WillSmith.jpg")
imgWill = cv2.cvtColor(imgWill,cv2.COLOR_BGR2RGB)

imgWillTest= face_recognition.load_image_file("WillSmith2.jpg")
imgWillTest =cv2.cvtColor(imgWillTest,cv2.COLOR_BGR2RGB)


faceLocation = face_recognition.face_locations(imgWill)[0]
encodeWill=face_recognition.face_encodings(imgWill)[0]
cv2.rectangle(imgWill,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(0,255,0),2)


faceLocationTest = face_recognition.face_locations(imgWillTest)[0]
encodeWillTest=face_recognition.face_encodings(imgWillTest)[0]
cv2.rectangle(imgWillTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(0,255,0),2)


results= face_recognition.compare_faces([encodeWill],encodeWillTest) # see if its the same person

faceDist= face_recognition.face_distance([encodeWill],encodeWillTest)
print(results,faceDist)

cv2.putText(imgWillTest,f'{results} {round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow('Will Smith',imgWill)
cv2.imshow('Will Smith Test',imgWillTest)

cv2.waitKey(0)







