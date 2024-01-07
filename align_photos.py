from PIL import Image
import pathlib
import cv2

PIC_PATH = 'pictures'
EXPORT_PATH = 'out'


FACE_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_frontalface_default.xml'
EYES_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier( str(FACE_CASCADE_PATH) )
eyes_cascade = cv2.CascadeClassifier( str(EYES_CASCADE_PATH) )


TARGET_EYE_LEFT: tuple[int, int]
TARGET_EYE_RIGHT: tuple[int, int]


for i, img_path in enumerate( pathlib.Path(PIC_PATH).iterdir() ):

    img = cv2.imread( str(img_path), 1)  # load image in grayscale

    # Faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(300, 300),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # loop over biggest face, so only executed if found one at all
    for (x1, y1, width_face, height_face) in sorted(list(faces), key=lambda f:f[2])[:1]:
        # cv2.rectangle( img, (x1, y1), (x1+width_face, y1+height_face), (255, 255, 0), 6 )

        # find Eyes
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

        source_eye_left: tuple[int, int]
        source_eye_right: tuple[int, int]

        # find both eyes, limit list to two eyes, don't know if they're ever more than two
        for j, (x2, y2, width_eye, height_eye) in enumerate(eyes[:2]):
            eye_center = ( x1 + x2 + width_eye // 2, y1 + y2 + height_eye // 2 )
            radius = int(round( width_eye / 2 ))

            # blindly assign eyes to variables and swap them if the right eye is more left than the left eye
            if j == 0:
                source_eye_left  = eye_center
            else:
                source_eye_right = eye_center

                if source_eye_right[0] < source_eye_left[0]:
                    source_eye_left, source_eye_right = source_eye_right, source_eye_left

            cv2.rectangle( img, (x1+x2, y1+y2), (x1+x2+width_eye, y1+y2+height_eye), (255, 255, 0), 6 )

        # set global target eye coordinates
        if i == 0:
            TARGET_EYE_LEFT  = source_eye_left
            TARGET_EYE_RIGHT = source_eye_right

        save_path = pathlib.Path(EXPORT_PATH) / f'{i}.jpg'

        pil_img: Image = Image.open( str(img_path) )
        pil_img = pil_img.rotate(90, expand=True)

        translate_vector: tuple[int, int] = ( TARGET_EYE_LEFT[0] - source_eye_left[0],
                                              TARGET_EYE_LEFT[1] - source_eye_left[1] )

        pil_img = pil_img.rotate(0, translate = translate_vector )  # move this left eye to the taget coordinates


        pil_img.save(save_path)
        pil_img.close()

        if i > 40:
            quit()







    cv2.imshow('img', cv2.resize(img, (388, 690) ))

    if cv2.waitKey( 0 ) == ord('q'):
        quit()

cv2.destroyAllWindows()

