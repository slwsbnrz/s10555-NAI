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

#pobierz dane do szkolenia i testowania:
dataset = datasets.load_digits()
#podziel dane na testowe i treningowe
(trainData, testData, trainLabels, testLabels) = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0)

#tworzy klasyfikator
clf = KNeighborsClassifier(n_neighbors=1)
 
# wlacz kamere
cam = cv2.VideoCapture(0)
char_mask = np.zeros((600, 480), dtype = "uint8")
 
# keep looping
while True:
	# przechwyc obraz
	#(grabbed, frame) = camera.read()
	ret, frame = cam.read()
 
	# zmienia wielkosc przechwyconego obrazu, zamazuje przy uzyciu funkcji gaussianblur
	# przeksztalca kolor z BGR do HSV
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
	# tworzy maske 
	
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
		# znajdz kontur
	# (x, y) oraz srodek kolka
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
 
		#narysuj kolo
		cv2.circle(frame, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
		cv2.circle(frame, center, 5, (0, 0, 255), -1)
 
	# dodaj punkty do kolejki
	pts.appendleft(center)
		# petla po znalezionych punktach
 
		#rysuj linie
	
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

		if key == ord(" "):
			char_mask = np.zeros((600, 480), dtype = "uint8")

			#rysuj linie na masce

			for i in xrange(1, len(pts)):
				
				cv2.line(char_mask, pts[i - 1], pts[i], (255, 255, 255), 5)

			#popraw
			cv2.dilate(character_mask, None, iterations=3)
			#wykryj zewnetrzne kontury, tylko punkty koncowe
			cnts, _ = cv2.findContours(char_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			#znajdz maksymalny kontur narysuj wokol niego prostokat
			cnt = max(cnts, key=cv2.contourArea)
			(x, y, w, h) = cv2.boundingRect(cnt)



 
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# zamyka okno konsoli przy nacisnieciu klawisza q
	if key == ord("q"):
		break
 
# wyczysc obraz z kamery i zamknij okno

cv2.destroyAllWindows()
