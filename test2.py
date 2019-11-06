#Paquetes necesarios
import imutils 
import argparse
import time
import cv2
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS





#Parseador de argumentos para seleccionar un video y el tipo de tracking
#OpenCV cuenta con varios como csrt, kcf, mosse, medianFlow etc... 
#Si no se introuce la ruta a un video existente, se procede a caputrar video 
#de un dispositivo de video (camara web)
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="Ruta del video")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="Tipo de Tracker")
args =  vars(ap.parse_args())

#Inicializacion de un diccionario que mapea strings con su opencv object tracker correspondiente
#BOOSTING Tracker: algoritmo utilizado detrás de las cascadas Haar (AdaBoost), tiene más de una década. Este rastreador es lento y no funciona muy bien, (mínimo OpenCV 3.0.0) 
#MIL Tracker: mejor precisión que BOOSTING tracker pero hace un mal trabajo al informar sobre fallas. (mínimo OpenCV 3.0.0) 
#KCF Tracker: filtros de correlación kernelized. Más rápido que BOOSTING y MIL. Similar a MIL y KCF, no maneja bien la oclusión completa. (mínimo OpenCV 3.1.0) 
#CSRT Tracker: filtro de correlación discriminativo (con canal y confiabilidad espacial). Tiende a ser más preciso que KCF pero un poco más lento. (mínimo OpenCV 3.4.2) 
#MedianFlow Tracker: hace un buen trabajo al informar fallas; sin embargo, si hay un salto demasiado grande en movimiento, como objetos que se mueven rápidamente u objetos que cambian rápidamente en su apariencia, el modelo fallará. (mínimo OpenCV 3.0.0) 
#Rastreador de TLD:  propenso a falsos positivos. No recomiendo usar este rastreador de objetos OpenCV. (mínimo OpenCV 3.0.0) 
#Rastreador MOSSE: Muy, muy rápido. No es tan preciso como CSRT o KCF, pero es una buena opción si necesita velocidad pura. (mínimo OpenCV 3.4.1)


#Control de versiones
(major, minor) = cv2.__version__.split(".")[:2]
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
    }
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None




if not args.get("video", False):
    print("[INFO] starting video Stream....")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
#Sino agarra el video de la ruta proporcionada
else:
    vs = cv2.VideoCapture(args["video"])
fps = None

#Loop sobre los cuadros del video stream
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame  = imutils.resize(frame, width=1000)
    retval, threshold = cv2.threshold(frame, 12, 255, cv2.THRESH_BINARY)
    greyscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval2, threshold2 = cv2.threshold(greyscaled, 12, 255, cv2.THRESH_BINARY)
    gaus = cv2.adaptiveThreshold(greyscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)



    #HSV para modificar la imagen de video y mostrar solamente colores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    lower_red = np.array([155,25,0])
    upper_red = np.array([255,255,255])
    mask = cv2.inRange(frame, lower_red, upper_red)
    #Mask0 = Rojo
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)
    #mask1 = Rojo[s]
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #finalmask
    finalmask = mask0+mask1
    
    finalmask = cv2.bitwise_and(frame, frame, mask = mask)
    (H,W) = mask.shape[:2]
    
    if initBB is not None:
        (success, box) = tracker.update(mask)
        if success:
            (x,y,w,h) = [int(v) for v in box]
            cv2.rectangle(mask, (x,y), (x+w, y+h), (0,255,0), 2)
        fps.update()
        fps.stop()
        info = [
            ("Tracker", args["tracker"]),
            ("Succes", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]  
        for (i, (k, v)) in enumerate(info):
            text = "{}:{}".format(k,v)
            cv2.putText(finalmask, text, (10, H - ((i*20)+20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

            
    cv2.imshow("Frame", frame)
    cv2.imshow("res", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", mask, fromCenter=False, showCrosshair=True)
        tracker.init(mask, initBB)
        fps = FPS().start()
    elif key == ord("q"):
        break
    
if not args.get("video", False):
    vs.stop()

else: 
    vs.release()
    cv2.destroyAllWindows()










