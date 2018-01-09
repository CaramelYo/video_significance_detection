import logging
import sys
import os
import numpy as np
from PIL import Image


# # logging setting
# logging.basicConfig(level = logging.DEBUG,
#                     format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
#                     datefmt = '%Y-%m-%d %H:%M',
#                     handlers = [logging.FileHandler('data_preprocessing.log', 'w', 'utf-8'), ])

# console = logging.StreamHandler(sys.stdout)
# console.setLevel(logging.DEBUG)
# console.setFormatter(logging.Formatter('%(message)s'))
# logging.getLogger('').addHandler(console)


class Video:
    def __init__(self, path):
        self.path = path

        # load origin frame
        self.ori_dir = os.path.join(path, Video.origin_dir)
        self.ori_imgs = []
        Video.get_imgs(self.ori_dir, self.ori_imgs, 'RGB')
        self.ori_imgs = np.asarray(self.ori_imgs)
        # [n, w, h, ch] ??
        logging.debug('img shape = %s' % str(self.ori_imgs.shape))
        # logging.debug('img = %s' % self.ori_imgs[0])
        

        # load saliency map
        self.s_dir = os.path.join(path, Video.saliency_dir)
        self.s_imgs = []
        Video.get_imgs(self.s_dir, self.s_imgs, 'L')
        self.s_imgs = np.asarray(self.s_imgs)
        self.s_imgs = np.expand_dims(self.s_imgs, axis = len(self.s_imgs.shape))
        # logging.debug('s_img shape = %s' % str(self.s_imgs.shape))
        # exit()

        # for name in os.listdir(self.s_dir):
        #     p = os.path.join(self.s_dir, name)
        #     if os.path.isfile(p):
        #         img = Image.open(p, 'r')

        #         # r, g, b = rgb_img.getpixel((0, 0))
        #         # img.show()
        #         # shape = (h, w, channel)
        #         # img_matrix = np.asarray(img.convert('RGB'))
        #         self.s_imgs.append(np.asarray(img.convert('RGB')))
        #     else:
        #         logging.debug('path = %s is not a file' % p)

    def get_imgs(p, l, ch):
        for name in os.listdir(p):
            pp = os.path.join(p, name)
            if os.path.isfile(pp):
                img = Image.open(pp, 'r')

                # r, g, b = rgb_img.getpixel((0, 0))
                # img.show()
                # shape = (h, w, channel)
                # img_matrix = np.asarray(img.convert('RGB'))
                l.append(np.asarray(img.convert(ch)))
            else:
                logging.debug('path = %s is not a file' % pp)

    saliency_dir = 'saliency'
    origin_dir = 'frame'