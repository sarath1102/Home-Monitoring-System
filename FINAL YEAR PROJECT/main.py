import cv2,time
import numpy as np
import face_recognition
from datetime import datetime
import os
from datetime import datetime
import os
from twilio.rest import Client

increment1=0
increment2=0

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def sendAlertTwilio(name):
    account_sid=os.environ['TWILIO_ACCOUNT_SID']='AC6f98fa3bb9adc5d5a1b9718febaaf675'
    auth_token=os.environ['TWILIO_AUTH_TOKEN']='810d30376b7be1de3456080c953ef2f4'
    target_number = '+91 94004 33524'
    source_number = '+19785435769'
    client = Client(account_sid, auth_token)
    message = client.messages.create(body=name+' has entered frame.', from_=source_number, to=target_number)
    print(message.sid)

def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markKnown(name,increment):
    with open('KnownList.csv', 'r+') as f:
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            ret,frame = cap.read()
            cv2.imwrite('DETECTED_KNOWN/face' + str(increment) + '.jpg',frame)
            now = datetime.now()
            dtString = now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')


def markUnknown(name,increment):
    with open('UnknownList.csv', 'r+') as f:
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            ret,frame = cap.read()
            cv2.imwrite('UNKNOWN_FACE/unknown_face' + str(increment) + '.jpg',frame)
        if name not in nameList:
            sendAlertTwilio(name)
            now = datetime.now()
            dtString = now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')
            

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS,model="hog")
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace,tolerance=0.5)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print('Detected Face distance: ',faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print("Detected Face: ",name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markKnown(name,increment1)
            increment1 = increment1 + 1
        else:
            name = "Unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markUnknown(name,increment2)
            #time.sleep(1)
            increment2 = increment2 + 1

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
