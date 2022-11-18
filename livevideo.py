#Snap-Live Livestreaming

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os

app = Flask(__name__)
socketioApp = SocketIO(app)


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


cap = cv2.VideoCapture(0)

def gen_frames():
   
    while True:

        #cv2.imshow('img',img) #display image

        
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
            
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result
        
            #if user pressed 'q' break
        if cv2.waitKey(1) == ord('q'): # 
            break

    cap.release() #turn off camera  
    cv2.destroyAllWindows() #close all windows


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    #Video streaming Home Page
    
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)


