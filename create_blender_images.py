import os
from PIL import Image


SOURCE_DIR = 'pictures'
TARGET_DIR = 'rotated'

images: list = os.listdir(SOURCE_DIR)

for i, filename in enumerate(images):
    img: Image = Image.open( SOURCE_DIR + '/' + filename )
    img = img.rotate(90, expand=True)
    img.save(TARGET_DIR + '/' + str(i) + filename[-4:])
    # img.save(TARGET_DIR + '/' + filename[-4:])
    img.close()

