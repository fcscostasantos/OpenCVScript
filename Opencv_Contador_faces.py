import cv2

webCamera = cv2.VideoCapture(2)
classificadorVideoFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    camera, frame = webCamera.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = classificadorVideoFace.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=8,  minSize=(25, 25))
    for(x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)

        contador = str(detecta.shape[0])

        cv2.putText(frame, contador, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Quantidade de Faces: " + contador, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #cv2.putText(frame, "Quantidade de Faces Detectadas " + str(detecta.shape[0]), (0, frame.shape[0] - 10),
                   #cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Video WebCamera", frame)

    if cv2.waitKey(1) == ord('f'):
        break

webCamera.release()
cv2.destroyAllWindows()
