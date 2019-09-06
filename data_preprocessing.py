import logging
import sys
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import copy


# # logging setting
# logging.basicConfig(level = logging.DEBUG,
#                     format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
#                     datefmt = '%Y-%m-%d %H:%M',
#                     handlers = [logging.FileHandler('data_preprocessing.log', 'w', 'utf-8'), ])

# console = logging.StreamHandler(sys.stdout)
# console.setLevel(logging.DEBUG)
# console.setFormatter(logging.Formatter('%(message)s'))
# logging.getLogger('').addHandler(console)


class Wound:
    def __init__(self, ori_dir, truth_dir):
        # self.path = path

        # load original img
        # self.ori_dir = os.path.join(path, Wound.original_dir)
        self.ori_dir = ori_dir
        self.ori_imgs = []
        self.ori_names = Wound.get_imgs(self.ori_dir, self.ori_imgs, 'RGB')
        self.ori_imgs = np.array(self.ori_imgs, dtype = np.float32)
        logging.info('imgs shape = %s' % str(self.ori_imgs.shape))
        
        # self.truth_dir = os.path.join(path, Wound.answer_dir)
        self.truth_dir = truth_dir
        self.truth_imgs = []
        self.truth_names = Wound.get_imgs(self.truth_dir, self.truth_imgs, 'L')
        self.truth_imgs = np.array(self.truth_imgs, np.float32)
        self.truth_imgs = np.expand_dims(self.truth_imgs, axis = len(self.truth_imgs.shape))

        # preprocess for truth_img so that the pixel range is from 0. to 1.
        # self.truth_imgs /= 255.
        self.truth_imgs = Wound.preprocessing(self.truth_imgs)

        logging.info('truth_imgs shape = %s' % str(self.truth_imgs.shape))

    def preprocessing(imgs, mode = '0-255_to_0-1'):
        if mode == '0-255_to_0-1':
            imgs /= 255.
        else:
            logging.error('unexpected preprocessing mode')
            exit()

        return imgs

    def postprocessing(imgs, mode = '0-1_to_0-255'):
        if mode == '0-1_to_0-255':
            # squeeze the last channel
            if imgs.shape[3] == 1:
                imgs = np.squeeze(imgs, axis = 3)
            
            imgs *= 255.
            imgs = imgs.astype(np.uint8)
        else:
            logging.error('unexpected postprocessing mode')
            exit()

        return imgs
    
    def get_img(path, imgs, ch, mag = 16, size_mag = 1):
        img = Image.open(path, 'r')

        w, h = img.size
        img = img.resize(((((w // size_mag) // mag) * mag),(((h // size_mag) // mag) * mag)))
        
        imgs.append(np.array(img.convert(ch)))
        return img

    def get_imgs(img_dir, imgs, ch, mag = 16, size_mag = 1):
        names = []

        for name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, name)
            if os.path.isfile(img_path):
                names.append(name)
                # img = Image.open(pp, 'r')

                # # r, g, b = rgb_img.getpixel((0, 0))
                # # img.show()
                # # shape = (h, w, channel)
                # # img_matrix = np.asarray(img.convert('RGB'))

                # w, h = img.size
                # img = img.resize((((w // size_mag) // mag) * mag, ((h // size_mag) // mag) * mag))

                # imgs.append(np.array(img.convert(ch)))
            
                Wound.get_img(img_path, imgs, ch, mag, size_mag)
            
            else:
                logging.error('path = %s is not a file' % img_path)
        
        return names
    
    def invert_imgs(imgs, names, out_dir, change_range = False):
        if change_range:
            new_imgs = imgs * 255.
        else:
            new_imgs = copy.copy(imgs)

        if new_imgs.shape[3] == 1:
            new_imgs = np.squeeze(new_imgs, axis = 3)
            ch = 'L'
        elif new_imgs.shape[3] == 3:
            ch = 'RGB'
        
        new_imgs = new_imgs.astype(np.uint8)
        out_imgs = []

        for i in range(new_imgs.shape[0]):
            img = Image.fromarray(new_imgs[i], mode = ch)
            out_img = ImageOps.invert(img)

            out_path = os.path.join(out_dir, names[i])

            out_img.save(out_path)

            out_imgs.append(np.array(out_img.convert(ch)))
        
        out_imgs = np.array(out_imgs, dtype = np.float32)

        if len(out_imgs.shape) == 3:
            out_imgs = np.expand_dims(out_imgs, axis = 3)

        return out_imgs, names

    def size_mag_imgs(imgs, names, size_mag, out_dir, change_range = False):
        if change_range:
            new_imgs = imgs * 255.
        else:
            new_imgs = copy.copy(imgs)
        
        if new_imgs.shape[3] == 1:
            new_imgs = np.squeeze(new_imgs, axis = 3)
            ch = 'L'
        elif new_imgs.shape[3] == 3:
            ch = 'RGB'
        
        new_imgs = new_imgs.astype(np.uint8)
        out_imgs = []

        for i in range(new_imgs.shape[0]):
            img = Image.fromarray(new_imgs[i], mode = ch)
            w, h = img.size
            out_img = img.resize((int(w * size_mag), int(h * size_mag)))

            out_path = os.path.join(out_dir, names[i])

            out_img.save(out_path)

            out_imgs.append(np.array(out_img.convert(ch)))

        out_imgs = np.array(out_imgs, dtype = np.float32)

        if len(out_imgs.shape) == 3:
            out_imgs = np.expand_dims(out_imgs, axis = 3)

        return out_imgs, names

    original_dir = 'origin'
    answer_dir = 'ground_truth'


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