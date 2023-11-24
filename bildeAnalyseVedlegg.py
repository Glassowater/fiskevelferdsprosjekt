
import cv2
import numpy as np

# Databehandling initialisering
import os
from google.cloud import firestore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Your firebase here" 

# Oppretter en Firestore-klient
db = firestore.Client()

dataset_count = 0

#Kamera og objektgjenkjenning

cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(history=2500, varThreshold=30)

red_speed = []
red_time = []
blue_speed = []
blue_time = []
lus_status = []
lus_tid = []

xr0, yr0, tr0 = 0, 0, cv2.getTickCount()
xb0, yb0, tb0 = 0, 0, cv2.getTickCount()

x, y, w, h = 0, 0, 0, 0
lus = 0

start_time = cv2.getTickCount()
check_time = start_time
calc_time = start_time
picture_time = cv2.getTickCount()

def colorFilter(img, hsv_lower_range, hsv_upper_range):
    
    #Lager hsv fargekart
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Lager bildet som er innenfor de spesifisierte fargene
    color_mask = cv2.inRange(hsv_img, hsv_lower_range, hsv_upper_range)
    color_image = cv2.bitwise_and(img, img, mask=color_mask)

    #Lager masken som tracker bildet
    mask = object_detector.apply(color_image)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

def findFish(img, contours, color, x0, y0):
    global red_picture_time, blue_picture_time
    t1 = cv2.getTickCount()
    n = 0
    h = 0
    w = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            if n == 0:
                x_small, y_small = x, y
                x_big, y_big = x, y
                n+=1
            x_small = min(x_small, x)
            y_small = min(y_small, y)
            x_big = max(x_big, x+w)
            y_big = max(y_big, y+h)
            
    if h > 0 or w > 0:
        cv2.rectangle(img,(x_small, y_small), (x_big, y_big), (0, 255, 0), 3)
        x_coor = (x_big-x_small)/2
        y_coor = (y_big-y_small)/2
        speed = calcSpeed(x0, y0, x_coor, y_coor)
        if color == "red" and speed < 400:
            red_speed.append(speed)
            red_time.append((t1-start_time)/cv2.getTickFrequency())
            red_picture_time = cv2.getTickCount() #Tar tiden på når sist rødt bilde ble tatt

        elif color == "blue" and speed < 400:
            blue_speed.append(speed)
            blue_time.append((t1-start_time)/cv2.getTickFrequency())
            blue_picture_time = cv2.getTickCount() #Tar tiden på når sist blått bilde ble tatt
    try:
        return x_coor, y_coor
    except:
        return x0, y0

def calcSpeed(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

def uploadData(red_speed, red_time, blue_speed, blue_time, lus_status, lus_tid):
    data = {
        "redFish" : red_speed,
        "redTime": red_time,
        "blueFish" : blue_speed,
        "blueTime" : blue_time
    }
    lus_data = {
        "lus_status" : lus_status,
        "lus_tid" : lus_tid
    }
    try:
        speed_doc = db.collection("Svømmemønster").document("fart"+ str(dataset_count))
        lus_doc = db.collection("Lus_data").document("lus_status" + str(dataset_count))
        speed_doc.set(data)
        lus_doc.set(lus_data)
        print("Data uploaded successfully")
    except Exception as e:
        print(f"Error adding data to Firestore: {e}")

    red_speed[:], red_time[:], blue_speed[:], blue_time[:], lus_status[:], lus_tid[:] = [], [], [], [], [], []

while True:
    ret, frame = cap.read() 
    now_time = cv2.getTickCount()
    #Bildeanalyse
    #RØDE MASKER
    red_mask, red_contours = colorFilter(frame, (179*0, 255*0.35, 255*0.2), (179*0.09, 255*1, 255*1)) #For rød fisk
    #red_mask, red_contours, red_pic = colorFilter(frame, (179*0, 255*0, 255*0), (179*0.09, 255*1, 255*1)) #For NEMO

    #BLÅ MASKER
    blue_mask, blue_contours = colorFilter(frame, (179*0.48, 255*0.25, 255*0.2), (179*0.67, 255*1, 255*1)) #For blå fisk
    #blue_mask, blue_contours, blue_pic = colorFilter(frame, (179*0.40, 255*0.6, 255*0.6), (179*0.7, 255*1, 255*0.9)) #For NEMO

    #GRØNNE MASKER 
    green_mask, g_contours = colorFilter(frame, (179*0.2, 50, 50), (179*0.35, 200, 200))  #For lus
    
    if (now_time-calc_time)/cv2.getTickFrequency() >= 1:
        xr0, yr0 = findFish(frame, red_contours, "red", xr0, yr0)
        xb0, yb0 = findFish(frame, blue_contours, "blue", xb0, yb0)
        for cnt in g_contours:
            g_area = cv2.contourArea(cnt)
            if g_area > 400:
                lus = 1
                break #Kan breake loopen fordi lus enten er 1 eller 0, så mengden lus har ikke noe å si, bare om det er > 0. Sparer processing power på dette
            lus = 0
        lus_status.append(lus)
        lus_tid.append((now_time-start_time)/cv2.getTickFrequency())
        calc_time = now_time
    #Sjekke om den skal sende data
    if (now_time-check_time)/cv2.getTickFrequency() > 30: #Laster opp dataene hvert 30. sekund
        check_time = cv2.getTickCount()
        uploadData(red_speed, red_time, blue_speed, blue_time, lus_status, lus_tid)
        dataset_count += 1


