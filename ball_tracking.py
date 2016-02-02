# importy dependencji
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
 
#definiuje dlugosc bufora
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
# definiuje przestrzen zieleni
# oraz inicjuje kolejke
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])
 
# wlacz kamere
camera = cv2.VideoCapture(0)
 
# keep looping
while True:
	# przechwyc obraz
	(grabbed, frame) = camera.read()
 
	# zmienia wielkosc przechwyconego obrazu, zamazuje przy uzyciu funkcji gaussianblur
	# przeksztalca kolor z BGR do HSV
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
	# tworzy maske 
	
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
		# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
 
	# jesli znaleziony jest minimum 1 kontur
	if len(cnts) > 0:
		# znajdz najwiekszy kontur
		# oblicz na podstawie momentow najmniejsze kolko
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# jesli srednica spelnia minimalne wymagania
		if radius > 10:
			# narysuj kolo
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
 
	# dodaj punkty do kolejki
	pts.appendleft(center)
		# petla po znalezionych punktach
	for i in xrange(1, len(pts)):
		# ignoruj punkty None w kolejce
		if pts[i - 1] is None or pts[i] is None:
			continue
 
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# zamyka okno konsoli przy nacisnieciu klawisza q
	if key == ord("q"):
		break
 
# wyczysc obraz z kamery i zamknij okno
camera.release()
cv2.destroyAllWindows()