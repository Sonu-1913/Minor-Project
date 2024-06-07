# This imports the OpenCV library which is used for computer vision and image processing.
import cv2

# This imports the NumPy library which is used for numerical computing and handling arrays.
import numpy as np

# This imports the face_recognition library which is a popular facial recognition library.

import face_recognition

# This imports the os library which is used for interacting with the operating system.
import os

# This imports the datetime module which is used for working with dates and times.
from datetime import datetime, timedelta

# This imports the smtplib module which is used for sending emails.
import smtplib

# This imports the MIMEMultipart class which is used for creating multipart email messages.
from email.mime.multipart import MIMEMultipart

# This imports the MIMEBase class which is used for creating email attachments.
from email.mime.base import MIMEBase

# This imports the MIMEText class which is used for creating text email messages.
from email.mime.text import MIMEText

# This imports the COMMASPACE constant which is used as a separator in email addresses.
from email.utils import COMMASPACE

# This imports the encoders module which is used for encoding email attachments.
from email import encoders

# Variables for attendance reset and email settings
last_reset = datetime.now()
fromaddr = "mirthintylohitaditya@gmail.com"
toaddr = "bp98488@gmail.com"


# function to Reset the CSV file
def reset_attendance():
    with open('listAttendance.csv', 'w') as f:
        f.write('')


# call reset_attendance for once in starting to keep it clean
reset_attendance()


# function to Send the attendance file to the preset email
def send_email():
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "TrainingImages File"
    body = "Please find the attached attendance file."
    msg.attach(MIMEText(body, 'plain'))
    filename = "listAttendance.csv"
    attachment = open(filename, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "xnuyzfqglnhrjiqm")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


# function to Check if one hour has passed since last reset
def check_reset_time():
    global last_reset
    if (datetime.now() - last_reset).seconds > 3600:
        send_email()
        reset_attendance()
        last_reset = datetime.now()


# function to Create path for trainingImages and create images list and classnames list
def load_images():
    path = "TrainingImages"
    images = []
    classNames = []
    for cl in os.listdir(path):
        curimg = cv2.imread(f'{path}/{cl}')
        images.append(curimg)
        classNames.append(os.path.splitext(cl)[0])
    encodeListKnown = find_encodings(images)
    return encodeListKnown, classNames


# function to Find encodings for given images
def find_encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


# function to Mark attendance of given name
def mark_attendance(name):
    with open('listAttendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            date_string = now.strftime('%Y-%m-%d')
            time_string = now.strftime('%H:%M:%S')
            f.write(f'{name},{date_string},{time_string}\n')


# function to Perform face recognition on a video stream captured by a webcam
def perform_face_recognition(encodeListKnown, classNames):
    cap = cv2.VideoCapture(0)
    while True:
        check_reset_time()
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    encodeListKnown, classNames = load_images()
    perform_face_recognition(encodeListKnown,classNames)