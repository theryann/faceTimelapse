from PIL import Image, ImageDraw
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

def find_face_coordinates(img: np.array):
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

def circle_from_three_points(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> tuple[int, int, float]:
    '''
    take three points as input and return x, y and radius of the circle they all lie upon.
    Uses formular A * b = c with b being the unnown.

    '''
    A = np.array([
        [2*a[0], 2*a[1], 1],
        [2*b[0], 2*b[1], 1],
        [2*c[0], 2*c[1], 1]
    ])
    c = np.array([
        -( a[0]**2 + a[1]**2 ),
        -( b[0]**2 + b[1]**2 ),
        -( c[0]**2 + c[1]**2 )
    ])
    try:
        b = np.linalg.solve(A, c)
    except:
        return None

    # calculate x, y and radious of circle
    x = -b[0]
    y = -b[1]
    r = np.sqrt( x**2 + y**2 - b[2] )

    return int(x), int(y), r

def find_pupil_center(img_eye: np.ndarray, offset_x, offest_y) -> tuple:
    '''
    finds the darkest pixels and returns the medium coordinate from them
    '''
    # create a list that holds the coordinates and luminancy of all the pixels
    luminace_list: list[tuple] = []
    for y in range(img_eye.shape[0]):
        for x in range(img_eye.shape[1]):
            avg = np.sum( img_eye[y][x] ) / 3
            luminace_list.append(
                (x, y, avg)
            )
            # img_eye[y][x] = np.array([avg,avg,avg])

    # store the darkest 10% of pixels in seperate array
    luminace_list.sort(key=lambda pix: pix[2], reverse=False)
    lowest_percentile = int(len(luminace_list) / 5)

    # calculate the medium x and y of the this array of darkest pixels

    medians_dark = np.median(
        np.array( luminace_list[:lowest_percentile] ),
        axis=0
    )
    medians_darker = np.median(
        np.array( luminace_list[:int(len(luminace_list) / 20)] ),
        axis=0
    )
    center_x = img_eye.shape[1] // 2
    center_y = img_eye.shape[0] // 2

    local_x = int(np.mean([ medians_dark[0], medians_dark[0], medians_dark[0], center_x, luminace_list[0][0] ]))
    local_y = int(np.mean([ medians_dark[1], medians_dark[1], center_y, luminace_list[0][1] ]))


    # use calculated midpoint as start to find the area of the pupil
    reduce_image = Image.fromarray(img_eye)
    reduce_image = reduce_image.quantize(colors = 50 ).convert('RGB')

    img_eye = np.array(reduce_image)

    pupil_pixels: set[tuple[int, int]] = set()
    threshold_percentage: float = 0.1

    anchor_luninance = np.median(np.array([
        img_eye[local_y-1][local_x-1], img_eye[local_y-1][local_x], img_eye[local_y-1][local_x+1],
        img_eye[local_y  ][local_x-1], img_eye[local_y  ][local_x], img_eye[local_y  ][local_x+1],
        img_eye[local_y+1][local_x-1], img_eye[local_y+1][local_x], img_eye[local_y+1][local_x+1],
    ]))

    def luminance_in_threshold(rgb: np.array):
        lum_pix = np.sum(rgb) / 3
        lum_anchor = anchor_luninance
        return abs(1 - lum_pix / lum_anchor  ) < threshold_percentage

    def find_dark_patch(x, y):
        if (x,y) in pupil_pixels:
            return

        if not luminance_in_threshold( img_eye[y][x] ):
            return

        pupil_pixels.add( (x, y) )

        try:
            find_dark_patch(x + 1, y )
            find_dark_patch(x - 1, y )
            find_dark_patch(x - 1, y + 1 )
            find_dark_patch(x - 1, y - 1 )
        except:
            return

    find_dark_patch(local_x, local_y) # start searching from first estimated eye center position

    while len(pupil_pixels) < 35 and threshold_percentage < 0.7:
        try:
            threshold_percentage += 0.1
            pupil_pixels.clear()
            find_dark_patch(local_x, local_y)
        except:
            break


    for x, y in pupil_pixels:
        img_eye[y][x] = np.array([255,255,0])


    med_x = int(np.nanmean([ coord[0] for coord in pupil_pixels ]))
    med_y = int(np.nanmean([ coord[1] for coord in pupil_pixels ]))

    # find three points from the lower third
    point_a = max(pupil_pixels, key=lambda pix: pix[1]) # lowest point
    point_c = max([pix for pix in pupil_pixels if med_y < pix[1] < point_a[1]], key=lambda pix: pix[1]) # pick most right point from points that are lower than the median
    point_b = min([pix for pix in pupil_pixels if med_y < pix[1] < point_a[1]], key=lambda pix: pix[1]) # pick most left point from points that are lower than the median

    cv2.circle(img_eye, point_a , radius= 2, color=(0,255,0), thickness=-1)
    cv2.circle(img_eye, point_b , radius= 2, color=(0,255,0), thickness=-1)
    cv2.circle(img_eye, point_c , radius= 2, color=(0,255,0), thickness=-1)

    res = circle_from_three_points(point_a, point_b, point_c)
    if res:
        x, y, r = res
        circle_center = x, y

        cv2.circle(img_eye, (med_x, med_y) , radius= 5, color=(255,0,0), thickness=-1)
        if 20 > np.linalg.norm( np.array(circle_center) - np.array((med_x, med_y)) ):
            midpoint = (np.array(circle_center) + np.array((med_x, med_y))) / 2
            med_x = int(midpoint[0])
            med_y = int(midpoint[1])
            cv2.circle(img_eye, (x, y) , radius= 5, color=(0,0,255), thickness=-1)

    if 20 > np.linalg.norm( np.array((center_x, center_y)) - np.array((med_x, med_y)) ):
        med_x = center_x
        med_y = center_y

    cv2.circle(img_eye, (med_x, med_y) , radius= 5, color=(0,255,255), thickness=-1)

    # cv2.imshow('t', img_eye)
    # if cv2.waitKey(0) == ord('q'):
    #     cv2.destroyAllWindows()
    #     quit()
    # cv2.destroyAllWindows()

    return med_x + offset_x, med_y + offest_y

def find_eye_coordinates(img: np.array, face: tuple):
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

        eyeROI = faceROI[
            eye_y + height_eye // 4: eye_y + (height_eye - height_eye // 4),
            eye_x: eye_x + width_eye
        ]

        darkest_point = find_pupil_center(eyeROI, face_x_start + eye_x, face_y_start + eye_y + height_eye // 4)

        # blindly assign eyes to variables and swap them if the right eye is more left than the left eye
        if j == 0:
            source_eye_left  = darkest_point
        else:
            source_eye_right = darkest_point

            if source_eye_right[0] < source_eye_left[0]:
                source_eye_left, source_eye_right = source_eye_right, source_eye_left

    return source_eye_left, source_eye_right

def align_image_by_eyes(img_path, source_eye_right, source_eye_left, ANCHOR_EYE_RIGHT, ANCHOR_EYE_LEFT, debug=False) -> Image:
    '''
    reads in current image, aligns it to the anchor image by scaling, translating and rotating
    so that the eyes of both images overlay
    returns this new image
    '''

    pil_img: Image = Image.open( str(img_path) )
    pil_img = pil_img.rotate(90, expand=True)  # ensure that the horizontal dimensions are recognized

    # get Vectors between the eyes to get angle between eye line
    a = np.array(source_eye_right) - np.array(source_eye_left) # Vector between the eyes of the current face
    b = np.array(ANCHOR_EYE_RIGHT) - np.array(ANCHOR_EYE_LEFT) # Vector between target eyes

    if debug:
        draw = ImageDraw.Draw(pil_img)
        draw.ellipse((source_eye_left[0]-5, source_eye_left[1]-5, source_eye_left[0]+5, source_eye_left[1]+5), 'green')
        draw.ellipse((source_eye_right[0]-5, source_eye_right[1]-5, source_eye_right[0]+5, source_eye_right[1]+5), 'green')

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
    return pil_img


def main(append=False, debug=False) -> None:
    global ANCHOR_DIMENSIONS

    PIC_PATH = 'pictures'
    EXPORT_PATH = 'out'


    ANCHOR_EYE_LEFT: tuple[int, int]
    ANCHOR_EYE_RIGHT: tuple[int, int]

    number_of_images = len(list(pathlib.Path(PIC_PATH).iterdir())) # number of images in the input directory
    number_already_stabilized = len(list(pathlib.Path(EXPORT_PATH).iterdir())) # numer of images in the out directory.

    for i, img_path in enumerate( pathlib.Path(PIC_PATH).iterdir() ):
        if ANCHOR_DIMENSIONS and i < 97:
            continue

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

        # TRANSFORM Image
        pil_img: Image = align_image_by_eyes(img_path, source_eye_right, source_eye_left, ANCHOR_EYE_RIGHT, ANCHOR_EYE_LEFT, debug=debug)

        # save new image
        save_path = pathlib.Path(EXPORT_PATH) / f'{i:0>5}.jpg'
        pil_img.save(save_path)
        pil_img.close()

        if debug:
            if i > 800:
                quit()

        print(f'\rface:\t{i}/{number_of_images}\t({round(i/number_of_images*100, 2) if i < number_of_images-1 else 100}%)' , end='\t\t\t')


if __name__ == '__main__':
    args = sys.argv[1:]

    if '-a' in args or '--append' in args:
        main(append=True)

    if '-d' in args or '--debug' in args:
        main(debug=True)

    else:
        main()
