import cv2

webCamera = cv2.VideoCapture(1)
classificadorVideoFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classificadorOlho = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

while True:
    camera, frame = webCamera.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = classificadorVideoFace.detectMultiScale(cinza)
    for(x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
        pegaOlho = frame[y:y + a, x:x + l]
        OlhoCinza = cv2.cvtColor(pegaOlho, cv2.COLOR_BGR2GRAY)
        localizaOlho = classificadorOlho.detectMultiScale(OlhoCinza)
        for (ox, oy, ol, oa) in localizaOlho:
            cv2.rectangle(pegaOlho, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

    cv2.imshow("Video WebCamera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webCamera.release()
cv2.destroyAllWindows()
