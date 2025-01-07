from PIL import Image
import numpy as np
import pathlib

class AverageFace:
    def __init__(self, stable_path: str):
        self.stable_path: str = stable_path
        self.pixel_matrix: np.array = None

    def prepare_empty_image(self, reference_img: np.array):
        self.pixel_matrix = np.zeros_like(reference_img, dtype=np.float64)

    def create_average_face_image(self):
        # cummulate number of images in case not the entire directory should be used
        # so len(directory) would be the wrong number
        number_of_images: int = 0

        # cummulativery add the rgb channels up
        for i, img_path in enumerate(pathlib.Path(self.stable_path).iterdir()):
            print('\r' + str(img_path), end='')

            # if i > 100:
            #     continue

            with Image.open( str(img_path) ) as pil_img:
                img: np.array = np.array( pil_img )

            if self.pixel_matrix is None:
                self.prepare_empty_image( img )

            self.pixel_matrix += img
            number_of_images  += 1

        # calc avery by dividing by amount of pictures
        self.pixel_matrix = ( self.pixel_matrix / number_of_images ).astype(np.uint8)

    def save_image(self, path: str=None):
        pil_image = Image.fromarray(self.pixel_matrix)
        pil_image.save(path)


if __name__ == '__main__':
    avg: AverageFace = AverageFace(stable_path='out')
    avg.create_average_face_image()
    avg.save_image('average.jpg')