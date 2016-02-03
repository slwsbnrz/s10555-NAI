# importy dependencji
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

dataset = datasets.load_digits()
 
#definiuje dlugosc bufora
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
# definiuje przestrzen zieleni
# oraz inicjuje kolejke

greenMix = (29, 86, 6)
greenMax = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

(trainData, testData, trainLabels, testLabels) = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0)

# stworz klasyfikator 
model = KNeighborsClassifier(n_neighbors=1)


#szkolenie klasyfikatora
model.fit(trainData, trainLabels)

cap = cv2.VideoCapture()
frame = imutils.resize(frame, width=600)

# obiekt do przechowywania litery
char = None

while True:

	# Przechwytywanie klatek
	ret, frame = cap.read()

	#uzycie funkcji blur
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# zmein BGR na HSV 
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#zastosowanie maski by wykryc tylko zielony kolor
	mask = cv2.inRange(hsv, greenMin, greenMax)

	# redukcja zaklocen
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# szukaj konturów obiektu
	contours= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	#zapisz srodek obiektu do center
	center = None
 
	# jesli wykryto przynajmniej 1 kontur
	if len(contours) > 0:

		#znajdz maksymalny kontur
		c = max(contours, key=cv2.contourArea)

		#wyznacz najmniejszy kontur przedmiotu
		(x, y) = cv2.minEnclosingCircle(c)
		
		# znajdz srodek obiektu
		M = cv2.moments(c)

		 # wyznacz srodek okregu za pomoca momentow
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
		# dodanie do kolejki srodka okregu	
		pts.appendleft(center)

		for i in xrange(1, len(pts)):

			## rysuj linie 
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key == ord("r"):
		pts.clear()

cv2.destroyAllWindows()
