import os
import time
import uuid
import cv2

IMAGES_PATH = r'C:/Users/Muharrem/OpenCVProjects/ip-projects/facedetection/data/images' 
number_images = 30
cap = cv2.VideoCapture(0)

for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# resimleri etiketle -> console'a !labelme yaz
# opendir -> data -> images. sonra -> file -> change output dir -> data -> labels. sonra file -> save automatically ac
# edit -> create rectangle -> yuzu sec



