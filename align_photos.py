import pathlib
import cv2

PIC_PATH = 'pictures'


FACE_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_frontalface_default.xml'
EYES_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_eye.xml'


face_cascade = cv2.CascadeClassifier( str(FACE_CASCADE_PATH) )
eyes_cascade = cv2.CascadeClassifier( str(EYES_CASCADE_PATH) )




for img_path in pathlib.Path(PIC_PATH).iterdir():

    img = cv2.imread( str(img_path), 1)  # load image in grayscale

    # Faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(300, 300),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x1, y1, width_face, height_face) in sorted(list(faces), key=lambda f:f[2])[:1]:
        # cv2.rectangle( img, (x1, y1), (x1+width_face, y1+height_face), (255, 255, 0), 6 )

        # EYES
        faceROI = img[y1:y1+height_face, x1:x1+width_face]

        eyes = eyes_cascade.detectMultiScale(
            faceROI,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(list(eyes)) != 2:
            continue

        for (x2, y2, width_eye, height_eye) in eyes:
            eye_center = ( x1 + x2 + width_eye // 2, y1 + y2 + height_eye // 2 )
            radius = int(round( width_eye / 2 ))

            cv2.rectangle( img, (x1+x2, y1+y2), (x1+x2+width_eye, y1+y2+height_eye), (255, 255, 0), 6 )



    cv2.imshow('img', cv2.resize(img, (388, 690) ))

    if cv2.waitKey( 0 ) == ord('q'):
        quit()

cv2.destroyAllWindows()

