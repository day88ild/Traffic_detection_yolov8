import numpy as np
import cv2 as cv
import random
from ultralytics import YOLO


def main():
	model = YOLO("runs/detect/train/weights/best.pt")

	color_dict = {0: (random.randint(0, 255),
					  random.randint(0, 255),
					  random.randint(0, 255)),
				  1: (random.randint(0, 255),
					  random.randint(0, 255),
					  random.randint(0, 255)),
				  2: (random.randint(0, 255),
					  random.randint(0, 255),
					  random.randint(0, 255)),
				  3: (random.randint(0, 255),
					  random.randint(0, 255),
					  random.randint(0, 255)),
				  4: (random.randint(0, 255),
					  random.randint(0, 255),
					  random.randint(0, 255))
				  }

	class_dict = {0: 'bic',  # bicycle
				  1: 'bus',
				  2: 'car',
				  3: 'mot',  # motorbike
				  4: 'per'  # person
				  }

	capture = cv.VideoCapture(0)

	while True:

		isTrue, im_pred = capture.read()

		im_pred = cv.resize(im_pred, (640, 640))

		results = model(im_pred)

		confs = results[0].boxes.conf
		bboxes = results[0].boxes.xyxy
		classes = results[0].boxes.cls

		for i in range(len(confs)):

			if confs[i] > 0.3:
				x1, y1, x2, y2 = bboxes[i]

				cv.rectangle(im_pred,
							 (int(x1), int(y1)), (int(x2), int(y2)),
							 color=color_dict[int(classes[i])],
							 thickness=2)
				cv.rectangle(im_pred,
							 (int(x1), int(y1) - 20), (int(x1) + 80, int(y1)),
							 color=color_dict[int(classes[i])],
							 thickness=-1)

				cv.putText(im_pred,
						   f"{class_dict[int(classes[i])]}-{round(float(confs[i]), 2)}",
						   (int(x1) + 5, int(y1) - 5),
						   cv.FONT_HERSHEY_SIMPLEX,
						   0.5,
						   (255, 255, 255),
						   1)

		cv.imshow("Image pred (press q to quit)", im_pred)

		if cv.waitKey(20) & 0xFF == ord("q"):
			break

	capture.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()
