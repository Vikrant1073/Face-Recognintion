from tkinter.constants import FALSE
import numpy as np
import pandas as pd
import os
import csv

import cv2
from time import sleep

csv.field_size_limit(2147483647)


def detect(sem,sec):
    
    if sem == '1' or sem == '2':
        year = 'first_year'
    elif sem == '3' or sem == '4':
        year = 'second_year'
    elif sem == '5' or sem == '6':
        year = 'third_year'
    else:
        year = 'fourth_year'

    filename = f'data/Attendance_xlsx/{year}_{sem}sem_IT{sec}.xlsx'

    fname = '../recognizer/trainingData.yml'
    if not os.path.isfile(fname):
        print('first train the data')
        exit(0)


    names = {}
    labels = []
    students = []


    def from_excel_to_csv():
        filename = f'data/Attendance_xlsx/{year}_{sem}sem_IT{sec}.xlsx'
        df = pd.read_excel(filename)
        df.to_csv('../data.csv')



    def getdata():
        with open('../data.csv','r') as f:
            data = csv.reader(f)
            next(data)
            lines = list(data)
            for line in lines:
                names[int(line[0])] = line[3]


    def  markPresent(name):
        with open('../data.csv','r') as f:
            data = csv.reader(f)
            lines = list(data)

            for line in lines:

                
                if line[3] == name:
                    line[6] = 'P'
               

                    with open('../data.csv','w') as g:
                        writer = csv.writer(g,lineterminator='\n')
                        writer.writerows(lines)
                    break



    def update_Excel(filename):
        with open('./data.csv') as f:
            
            data = csv.reader(f)
            lines = list(data)
            
            with open('./data.csv','w') as g:
                writer = csv.writer(g,lineterminator='\n')
                
                writer.writerow(lines)
                
        df = pd.read_csv(r'../data.csv')

        filename = f'data/Attendance_xlsx/{year}_{sem}sem_IT{sec}.xlsx'
        df.to_excel(filename,index = False)
 
        
        print('Attendance is marked in excel')
        

    face_cascade = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture('../test_videos/5.mp4')


    from_excel_to_csv()
    getdata() 
    print('Total students :',names)

    recognizer =cv2.face.LBPHFaceRecognizer_create() #LOCAL BINARY PATTERNS HISTOGRAMS Face Recognizer

    recognizer.read(fname) #read the trained yml file

    num=0
    while True:   
        ret, img = cap.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray) 
        final = cv2.medianBlur(equ, 3)

        faces = face_cascade.detectMultiScale(final, 1.3, 8)
        

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            faceid, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print('label:',faceid)
            print('confidence:',confidence)
            predicted_name=names[faceid]
            
            
            if confidence < 80:
                confidence = 100 - round(confidence)
                

                
                cv2.putText(img, predicted_name , (x+2,y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,0),2)
                labels.append(faceid)
                students.append(names[faceid])
                totalstudents = set(students)
                justlabels = set(labels)
                print('student Recognised : ',totalstudents,justlabels)
                for i in justlabels:
                    if labels.count(i)>20:
                        markPresent(names[faceid])
    
            

            
            
            f=1
            cv2.imshow('Face Recognizer',img)
            k = cv2.waitKey(100) & 0xff
            
            num+=1
            if num>100:
                cap.release()
                sleep(10)
                print('we are done!')
                f=0
                break
        
        if f==0:
            update_Excel(filename)
            cv2.destroyAllWindows()
            break   