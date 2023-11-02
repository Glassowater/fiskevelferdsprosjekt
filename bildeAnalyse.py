
import cv2
import numpy as np

# Databehandling initialisering
import os
from google.cloud import firestore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Your_Firebase_Here"
# Opprett en Firestore-klient
db = firestore.Client()

dataset_count = 0

#Kamera og objektgjenkjenning

cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(history=2500, varThreshold=30)

red_speed = []
red_time = []
blue_speed = []
blue_time = []

xr0, yr0, tr0 = 0, 0, cv2.getTickCount()
xb0, yb0, tb0 = 0, 0, cv2.getTickCount()

x, y, w, h = 0, 0, 0, 0

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
    return mask, contours, color_image

def findFish(img, contours, color, x0, y0):
    global red_picture_time, blue_picture_time
    t1 = cv2.getTickCount()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            #cv2.drawontours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x, y), (x + w, y + h), (0, 255, 0), 3)
            t1 = cv2.getTickCount()
            speed = calcSpeed(x0, y0, x, y)
            if color == "red" and speed < 400:
                red_speed.append(speed)
                red_time.append((t1-start_time)/cv2.getTickFrequency())
                red_picture_time = cv2.getTickCount() #Tar tiden på når sist blått bilde ble tatt

            elif color == "blue" and speed < 400:
                blue_speed.append(speed)
                blue_time.append((t1-start_time)/cv2.getTickFrequency())
                blue_picture_time = cv2.getTickCount() #Tar tiden på når sist blått bilde ble tatt
    try:
        return x, y
    except:
        return x0, y0

def calcSpeed(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

def uploadData(red_speed, red_time, blue_speed, blue_time):
    data = {
        "redFish" : red_speed,
        "redTime": red_time,
        "blueFish" : blue_speed,
        "blueTime" : blue_time
    }
    try:
        speed_doc = db.collection("Svømmemønster").document("fart"+ str(dataset_count))
        speed_doc.set(data)
        print(data)
        print("Data uploaded successfully")
    except Exception as e:
        print(f"Error adding data to Firestore: {e}")

    red_speed = []
    red_time = []
    blue_speed = []
    blue_time = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    now_time = cv2.getTickCount()

    #Bildeanalyse
    red_mask, red_contours, red_pic = colorFilter(frame, (179*0, 255*0.3, 255*0.2), (179*0.09, 255*1, 255*0.7))

    blue_mask, blue_contours, blue_pic = colorFilter(frame, (179*0.48, 255*0.2, 255*0.2), (179*0.67, 255*1, 255*0.60))

    if now_time-calc_time >= 1:
        xr0, yr0 = findFish(frame, red_contours, "red", xr0, yr0)

        xb0, yb0 = findFish(frame, blue_contours, "blue", xb0, yb0)

        calc_time = now_time

    #Sjekke om den skal sende data

    if (now_time-check_time)/cv2.getTickFrequency() > 30: #Laster opp dataene hvert 30. sekund
        check_time = cv2.getTickCount()
        uploadData(red_speed, red_time, blue_speed, blue_time)
        dataset_count += 1

    cv2.imshow("frame", frame)
    cv2.imshow("red", red_pic)
    cv2.imshow("blue", blue_pic)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
