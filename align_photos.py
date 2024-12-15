from PIL import Image
import pathlib
import cv2
import numpy as np
import sys

ANCHOR_DIMENSIONS: tuple[int, int] = tuple()

FACE_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_frontalface_default.xml'
EYES_CASCADE_PATH = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier( str(FACE_CASCADE_PATH) )
eyes_cascade = cv2.CascadeClassifier( str(EYES_CASCADE_PATH) )


def scale_image(image: Image, factor: float) -> Image:
    '''
    create new image with target dimensions. then paste the up- or downscaled image into the new.
    this avoids rounding errors that result in widths/heights being off by one pixel.
    '''

    res_image: Image = Image.new('RGB', ANCHOR_DIMENSIONS)

    # scale the image
    image = image.resize((
        int(image.size[0] * factor),
        int(image.size[1] * factor)
    ))

    if factor > 1:
        # crop upscaled image to the target dimensions
        h, v = (image.size[0] - ANCHOR_DIMENSIONS[0]) // 2, (image.size[1] - ANCHOR_DIMENSIONS[1]) // 2

        image = image.crop(( h, v, image.size[0] - h, image.size[1] - v ))

    res_image.paste( image )
    return res_image

def calc_new_eye_pos(scaling_factor: float, old_pos: tuple) -> tuple:
    '''
    after scaling the image the position of the eyes change.
    this functions calculates the new position of the eyes (or any other point) after the scaling.
    '''
    new_pos: np.ndarray = np.array(old_pos) * scaling_factor

    if scaling_factor > 1:
        h, v = tuple(
            ( np.array(ANCHOR_DIMENSIONS) * scaling_factor - np.array(ANCHOR_DIMENSIONS) ) // 2
        )

        new_pos -= np.array([ h, v ])

    return tuple( new_pos.astype(int) )

def find_face_coordinates(img: Image):
    '''
    returns the room of interest (x, y, w, h) of the biggest detected face
    or None if no face found
    '''
    global face_cascade

    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(300, 300),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    found_faces = sorted(list(faces), key=lambda f:f[2])[:1]

    if len(found_faces) == 0:
        return None

    return found_faces[0]

def find_eye_coordinates(img: Image, face: tuple):
    '''
    returns the coordinates of the eyes as tuples or None of no eyes detected
    '''

    face_x_start, face_y_start, width_face, height_face = face

    # find Eyes
    faceROI = img[
        face_y_start: face_y_start + height_face,
        face_x_start: face_x_start + width_face
    ]

    eyes = eyes_cascade.detectMultiScale(
        faceROI,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(list(eyes)) != 2:
        return None

    source_eye_left: tuple[int, int]
    source_eye_right: tuple[int, int]

    # find both eyes, limit list to two eyes, don't know if they're ever more than two
    for j, (eye_x, eye_y, width_eye, height_eye) in enumerate(eyes[:2]):
        eye_center = ( face_x_start + eye_x + width_eye // 2, face_y_start + eye_y + height_eye // 2 )

        # blindly assign eyes to variables and swap them if the right eye is more left than the left eye
        if j == 0:
            source_eye_left  = eye_center
        else:
            source_eye_right = eye_center

            if source_eye_right[0] < source_eye_left[0]:
                source_eye_left, source_eye_right = source_eye_right, source_eye_left

        cv2.rectangle( img, (face_x_start+eye_x, face_y_start+eye_y), (face_x_start+eye_x+width_eye, face_y_start+eye_y+height_eye), (255, 255, 0), 6 )

    return source_eye_left, source_eye_right

def main(append=False) -> None:
    global ANCHOR_DIMENSIONS

    PIC_PATH = 'pictures'
    EXPORT_PATH = 'out'


    ANCHOR_EYE_LEFT: tuple[int, int]
    ANCHOR_EYE_RIGHT: tuple[int, int]

    number_of_images = len(list(pathlib.Path(PIC_PATH).iterdir())) # number of images in the input directory
    number_already_stabilized = len(list(pathlib.Path(EXPORT_PATH).iterdir())) # numer of images in the out directory.

    for i, img_path in enumerate( pathlib.Path(PIC_PATH).iterdir() ):

        if append:
            if 0 < i < number_already_stabilized - 1:
                print(f'\rface:\t{i}/{number_of_images}\t({round(i/number_of_images*100, 2) if i < number_of_images-1 else 100}%)' , end='\t\t\t')
                continue

        img = cv2.imread( str(img_path), 1)  # load image in grayscale

        face = find_face_coordinates(img)
        if face is None:
            continue

        eyes = find_eye_coordinates(img, face)
        if eyes is None:
            continue

        source_eye_left:  tuple[int, int] = eyes[0]
        source_eye_right: tuple[int, int] = eyes[1]

        # set global target eye coordinates
        if not ANCHOR_DIMENSIONS:
            ANCHOR_EYE_LEFT  = source_eye_left
            ANCHOR_EYE_RIGHT = source_eye_right
            ANCHOR_DIMENSIONS = img.shape[1], img.shape[0] # swap order because of vertical mode

        ###################
        # TRANSFORM Image #
        ###################

        pil_img: Image = Image.open( str(img_path) )
        pil_img = pil_img.rotate(90, expand=True)  # ensure that the horizontal dimensions are recognized

        # get Vectors between the eyes to get angle between eye line
        a = np.array(source_eye_right) - np.array(source_eye_left) # Vector between the eyes of the current face
        b = np.array(ANCHOR_EYE_RIGHT) - np.array(ANCHOR_EYE_LEFT) # Vector between target eyes


        # scale face to match target
        face_scaling_factor: float = 1 / ( np.linalg.norm(a) / np.linalg.norm(b) ) # ratio of current face to target face
        pil_img = scale_image( pil_img, face_scaling_factor )


        # rotate face to right angle
        new_left_eye  = calc_new_eye_pos(face_scaling_factor, source_eye_left)
        new_right_eye = calc_new_eye_pos(face_scaling_factor, source_eye_right)

        # vector to move the face so that the eyes align
        translate_vector: tuple = tuple( np.array(ANCHOR_EYE_LEFT) - np.array(new_left_eye) )

        # angle to rotate the face by
        angle_deg: float = np.degrees( np.arccos(
            np.dot(a, b) / ( np.linalg.norm(a) * np.linalg.norm(b) )
        ))

        final_right_eye = tuple( np.array(new_right_eye) + np.array(translate_vector) )
        if final_right_eye[1] < ANCHOR_EYE_RIGHT[1]:
            angle_deg = -angle_deg

        pil_img = pil_img.rotate(
            angle_deg,
            Image.BICUBIC,
            center=new_left_eye,
            translate=translate_vector
        )


        # save new image
        save_path = pathlib.Path(EXPORT_PATH) / f'{i}.jpg'
        pil_img.save(save_path)
        pil_img.close()

    print(f'\rface:\t{i}/{number_of_images}\t({round(i/number_of_images*100, 2) if i < number_of_images-1 else 100}%)' , end='\t\t\t')


if __name__ == '__main__':
    args = sys.argv[1:]

    if '-a' in args or '--append' in args:
        main(append=True)

    else:
        main()
