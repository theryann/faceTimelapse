from PIL import Image, ImageDraw
import pathlib
import cv2
import numpy as np

PIC_PATH = 'pictures'
EXPORT_PATH = 'out'


FACE_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_frontalface_default.xml'
EYES_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier( str(FACE_CASCADE_PATH) )
eyes_cascade = cv2.CascadeClassifier( str(EYES_CASCADE_PATH) )


TARGET_EYE_LEFT: tuple[int, int]
TARGET_EYE_RIGHT: tuple[int, int]
TARGET_DIMENSIONS: tuple[int, int]


def scale_image(image: Image, factor: float) -> Image:
    '''
    create new image with target dimensions. then paste the up- or downscaled image into the new.
    this avoids rounding errors that result in widths/heights being off by one pixel.
    '''

    res_image: Image = Image.new('RGB', TARGET_DIMENSIONS)

    # scale the image
    image = image.resize(
        ( int(image.size[0] * factor), int(image.size[1] * factor) )
    )

    if factor > 1:
        # crop upscaled image to the target dimensions
        h, v = (image.size[0] - TARGET_DIMENSIONS[0]) // 2, (image.size[1] - TARGET_DIMENSIONS[1]) // 2

        image = image.crop(( h, v, image.size[0] - h, image.size[1] - v ))

    res_image.paste( image )
    return res_image

def calc_new_eye_pos(scaling_factor: float, old_pos: tuple) -> tuple:
    '''
    after scaling the image the position of the eyes change.
    this functions calculates the new position of (the left) eye on which the remaining transformation base.
    '''
    new_pos: np.ndarray = np.array(old_pos) * scaling_factor

    if scaling_factor > 1:
        h, v = tuple(
            ( np.array(TARGET_DIMENSIONS) * scaling_factor - np.array(TARGET_DIMENSIONS) ) // 2
        )

        new_pos -= np.array([ h, v ])

    return tuple( new_pos.astype(int) )




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
            TARGET_DIMENSIONS = img.shape[1], img.shape[0] # swap order because of vertical mode

        ###################
        # TRANSFORM Image #
        ###################

        pil_img: Image = Image.open( str(img_path) )
        pil_img = pil_img.rotate(90, expand=True)  # ensure that the horizontal dimensions are recognized

        translate_vector: tuple[int, int] = ( TARGET_EYE_LEFT[0] - source_eye_left[0],
                                              TARGET_EYE_LEFT[1] - source_eye_left[1] )

        # get Vectors between the eyes to get angle between eye line
        a = (np.array(source_eye_right) + np.array(translate_vector)) - np.array(TARGET_EYE_LEFT) # Vector between left eye and this faces right eye
        b =  np.array(TARGET_EYE_RIGHT) - np.array(TARGET_EYE_LEFT) # Vector between left eye and target faces right eye


        # scale face to match target
        face_scaling_factor: float = 1 / (np.linalg.norm(a) / np.linalg.norm(b)) # ratio of current face to target face
        pil_img = scale_image( pil_img, face_scaling_factor )




        # rotate face to right angle


        angle_deg: float = np.degrees( np.arccos(
            np.dot(a, b) / ( np.linalg.norm(a) * np.linalg.norm(b) )
        ))

        if source_eye_right[1] > TARGET_EYE_RIGHT[1]:
            angle_deg *= -1

        # pil_img = pil_img.rotate( angle_deg, Image.BILINEAR, center=TARGET_EYE_LEFT )

        start = calc_new_eye_pos(face_scaling_factor, source_eye_left)
        end = calc_new_eye_pos(face_scaling_factor, source_eye_right)


        draw = ImageDraw.Draw(pil_img)
        draw.line( [ source_eye_left, source_eye_right ], fill='blue', width=10 )
        draw.line( [ start, end ], fill='red', width=10 )




        # save new image
        save_path = pathlib.Path(EXPORT_PATH) / f'{i}.jpg'
        pil_img.save(save_path)
        pil_img.close()

        if i > 18:
            quit()







    # cv2.imshow('img', cv2.resize(img, (388, 690) ))

    # if cv2.waitKey( 0 ) == ord('q'):
    #     quit()

# cv2.destroyAllWindows()

