import cv2

webcamera = cv2.VideoCapture(1)

while True:
    camera, frame = webcamera.read()

    cv2.imshow("Imagem WebCamera", frame)

    if cv2.waitKey(1) == ord('f'):
        break

webcamera.release()
cv2.destroyAllWindows()