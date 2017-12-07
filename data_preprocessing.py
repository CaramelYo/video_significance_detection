import logging
import sys
import os
# import pillow
from PIL import Image


# logging setting
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('data_preprocessing.log', 'w', 'utf-8'), ])

class Video_Data:
    def __init__(self, path):
        img = Image.open(path, 'r')
        rgb_img = img.convert('RGB')
        r, g, b = rgb_img.getpixel((0, 0))

        logging.debug(r, g, b)
