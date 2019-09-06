import logging
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import math
import time
from sklearn.model_selection import KFold, train_test_split
import tkinter as tk
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename

from data_preprocessing import Wound
from FCN import FCN16


# logging setting
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('main.log', 'w', 'utf-8'), ])

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

# data
data_dir = 'data'
vgg_path = os.path.join(data_dir, 'vgg16.npy')
wound_dir = os.path.join(data_dir, 'wound')
wound_training_dir = os.path.join(wound_dir, 'training')
# wound_validation_dir = os.path.join(wound_dir, 'validation')
wound_test_dir = os.path.join(wound_dir, 'test')
wound_result_dir = os.path.join(wound_dir, 'result')
original_dir = 'origin'
truth_dir = 'ground_truth'
wound_training_original_dir = os.path.join(wound_training_dir, original_dir)
wound_training_truth_dir = os.path.join(wound_training_dir, truth_dir)
wound_test_original_dir = os.path.join(wound_test_dir, original_dir)
wound_test_truth_dir = os.path.join(wound_test_dir, truth_dir)


# network setting
training_epochs = 2
display_epoch = 2
save_epoch = 1

batch_size = 2
display_batch = 2

train_max_iter = 3
valid_max_iter = 1

model_dir = os.path.join(data_dir, 'wound_model')
model_name = 'fcn'
model_log_dir = os.path.join(data_dir, 'wound_model_log')

# interface setting
img_mag = 0.5

# global imgs
ori_imgs = None
ori_img_list = []
ori_img_paths = []
ori_img_ch = 'RGB'

truth_imgs = None
truth_img_list = []
truth_img_paths = []
truth_img_ch = 'L'

predicted_imgs = None

result_imgs = None


def get_total_batch(y):
    # # floor
    # return int(y.shape[0] / batch_size)

    # ceil
    return math.ceil(y.shape[0] / batch_size)

def train(n_splits = 10):
    # load training wound data
    training_data = Wound(wound_training_original_dir, wound_training_truth_dir)

    fcn = FCN16(vgg_path)
    optimizer, accuracy, loss = fcn.build(x, y, is_train = True)

    # create a saver
    saver = tf.train.Saver()

    # start training
    with tf.Session() as sess:
        # for tensorboard
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(model_log_dir, graph = sess.graph)

        if os.path.isdir(model_dir):
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            logging.info('restoring model is completed')
        else:
            logging.info('training start!')
            # encounter the size problem ???
            g_init = tf.global_variables_initializer()
            sess.run(g_init)
        
        l_init = tf.local_variables_initializer()
        sess.run(l_init)

        # get total batches
        total_batches = []
        total_batches.append(get_total_batch(training_data.truth_imgs))
        logging.info('training wounds has total_batch = %d' % total_batches[0])

        epoch = 0
        while epoch < training_epochs:
            for i in range(len(total_batches)):
                # n_fold
                k_fold = KFold(n_splits = n_splits)
                split = 0

                for train_indices, valid_indices in k_fold.split(training_data.truth_imgs):
                    train_set_x = []
                    train_set_y = []
                    valid_set_x = []
                    valid_set_y = []

                    for index in train_indices:
                        train_set_x.append(training_data.ori_imgs[index])
                        train_set_y.append(training_data.truth_imgs[index])

                    train_set_x = np.array(train_set_x, dtype = np.float32)
                    train_set_y = np.array(train_set_y, dtype = np.float32)

                    for index in valid_indices:
                        valid_set_x.append(training_data.ori_imgs[index])
                        valid_set_y.append(training_data.truth_imgs[index])

                    valid_set_x = np.array(valid_set_x, dtype = np.float32)
                    valid_set_y = np.array(valid_set_y, dtype = np.float32)

                    # train_set_x = np.array([training_data.ori_imgs[index] for index in train_indices])
                    # train_set_y = np.array([training_data.truth_imgs[index] for index in )

                    # get total batch
                    train_total_batch = get_total_batch(train_set_y)
                    logging.info('train_total_batch = %d' % train_total_batch)

                    valid_total_batch = get_total_batch(valid_set_y)
                    logging.info('valid_total_batch = %d' % valid_total_batch)

                    # count the step
                    t1 = epoch * n_splits * (train_max_iter * train_total_batch + valid_max_iter * valid_total_batch)
                    t2 = split * (train_max_iter * train_total_batch + valid_max_iter * valid_total_batch)
                    tt = t1 + t2

                    for j in range(train_max_iter):
                        t3 = j * train_total_batch
                        ttt = tt + t3
                        for k in range(train_total_batch):
                            # get the batches of x and y
                            batch_x = train_set_x[k * batch_size: (k + 1) * batch_size]
                            batch_y = train_set_y[k * batch_size: (k + 1) * batch_size]

                            # test whether data has run out
                            if batch_y.shape[0] != batch_size:
                                logging.error('data has run out')
                            
                            _, loss_val = sess.run([optimizer, loss], feed_dict = {
                                x: batch_x,
                                y: batch_y
                            })

                            logging.info('loss val in epoch %d split %d train_iter %d train_batch %d = %f' % (epoch, split, j, k, loss_val))

                            if k % display_batch == 0:
                                logging.info('accuracy in epoch %d split %d train_iter %d train_batch %d = %f' %
                                            (epoch, split, j, k, sess.run(accuracy, feed_dict = {
                                                x: batch_x,
                                                y: batch_y
                                            })))
                            
                            summary_str = sess.run(merged_summary_op, feed_dict = {
                                x: batch_x,
                                y: batch_y
                            })

                            summary_writer.add_summary(summary_str, ttt + k)

                    for j in range(0, valid_total_batch, display_batch):
                        batch_x = valid_set_x[j * batch_size: (j + 1) * batch_size]
                        batch_y = valid_set_y[j * batch_size: (j + 1) * batch_size]

                        logging.info('original accuracy in epoch %d split %d valid_batch %d = %f' %
                                    (epoch, split, j, sess.run(accuracy, feed_dict = {
                                        x: batch_x,
                                        y: batch_y
                                    })))

                    for j in range(valid_max_iter):
                        t3 = train_max_iter * train_total_batch + j * valid_total_batch
                        ttt = tt + t3
                        for k in range(valid_total_batch):
                            # get the batches of x and y
                            batch_x = valid_set_x[k * batch_size: (k + 1) * batch_size]
                            batch_y = valid_set_y[k * batch_size: (k + 1) * batch_size]

                            # test whether data has run out
                            if batch_y.shape[0] != batch_size:
                                logging.error('data has run out')
                            
                            _, loss_val = sess.run([optimizer, loss], feed_dict = {
                                x: batch_x,
                                y: batch_y
                            })

                            logging.info('loss val in epoch %d split %d valid_iter %d valid_batch %d = %f' % (epoch, split, j, k, loss_val))

                            if k % display_batch == 0:
                                logging.info('accuracy in epoch %d split %d valid_iter %d valid_batch %d = %f' %
                                            (epoch, split, j, k, sess.run(accuracy, feed_dict = {
                                                x: batch_x,
                                                y: batch_y
                                            })))
                            
                            summary_str = sess.run(merged_summary_op, feed_dict = {
                                x: batch_x,
                                y: batch_y
                            })

                            summary_writer.add_summary(summary_str, ttt + k)
                    
                    logging.info('a fold is completed')
                    split += 1

            if epoch % save_epoch == 0:
                saver.save(sess, os.path.join(model_dir, model_name), global_step = epoch)
                logging.info('save the model file %s in %s' % (model_name, model_dir))

            epoch += 1

    logging.info('training has done')

def predict(data_x, data_y = None):
    fcn = FCN16(vgg_path)
    accuracy = fcn.build(x, y)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        logging.info('restoring model is completed')

        l_init = tf.local_variables_initializer()
        sess.run(l_init)

        total_batch = get_total_batch(data_x)

        # get the prediction_tensor
        prediction_tensor = tf.get_collection('prediction_tensor')[0]

        predictions = None

        for i in range(total_batch):
            batch_x = data_x[i * batch_size: (i + 1) * batch_size]

            # test whether data_x has run out
            if batch_x.shape[0] != batch_size:
                logging.error('data_x has run out')
            
            # predict
            predicted_y = sess.run(prediction_tensor, feed_dict = {
                x: batch_x
            })

            if type(predictions) == np.ndarray:
                predictions = np.concatenate((predictions, predicted_y), axis = 0)
            else:
                predictions = predicted_y

            if type(data_y) == np.ndarray:
                batch_y = data_y[i * batch_size: (i + 1) * batch_size]

                logging.info('accuracy in batch %d in = %f' % (i, sess.run(accuracy, feed_dict = {
                    x: batch_x,
                    y: batch_y
                })))

    logging.info('prediction has been over')
    return predictions

def dc(g_img, r_img):
    global dc_text

    g_pixels = 0
    r_pixels = 0
    inter_pixels = 0

    for y in range(g_img.shape[0]):
        for x in range(g_img.shape[1]):
            if g_img[y][x] == 1.:
                g_pixels += 1
            
            if r_img[y][x] == 255:
                r_pixels += 1
            
            if g_img[y][x] == 1. and r_img[y][x] == 255:
                inter_pixels += 1

    dc = 2. * inter_pixels / (g_pixels + r_pixels)

    text = 'g_pixels = %d' % g_pixels + '\n'
    text += 'r_pixels = %d' % r_pixels + '\n'
    text += 'inter_pixels = %d' % inter_pixels + '\n'
    text += 'dc value = %f' % dc + '\n'

    dc_text.set(text)
    return dc

def print_img_interface(img, text, row, column):
    global img_mag, win

    w, h = img.size
    img = img.resize((int(w * img_mag), int(h * img_mag)))

    render = ImageTk.PhotoImage(img)

    img_box = tk.Label(win, image = render)
    img_box.image = render
    img_box.grid(row = row, column = column)

    img_label = tk.Label(win, text = text)
    img_label.grid(row = row + 1, column = column)

    return img_box, img_label

def load_ori_img_click():
    global ori_imgs, ori_img_list, ori_img_paths, ori_img_ch

    path = askopenfilename()
    ori_img_paths.append(path)
    img = Wound.get_img(path, ori_img_list, ori_img_ch)

    ori_img_box, ori_img_label = print_img_interface(img, 'original img', 1, 0)

    del ori_imgs
    ori_imgs = np.array(ori_img_list, dtype = np.float32)

def load_truth_img_click():
    global truth_imgs, truth_img_list, truth_img_paths, truth_img_ch

    path = askopenfilename()
    truth_img_paths.append(path)
    img = Wound.get_img(path, truth_img_list, truth_img_ch)

    truth_img_box, truth_img_label = print_img_interface(img, 'ground truth img', 1, 1)

    del truth_imgs
    truth_imgs = np.array(truth_img_list, dtype = np.float32)
    truth_imgs = Wound.preprocessing(truth_imgs)

def predict_img_click():
    global ori_imgs, truth_imgs, predicted_imgs
    global time_text

    t0 = time.time()

    predicted_imgs = predict(ori_imgs, truth_imgs)

    logging.info('predicted_imgs shape = %s' % str(predicted_imgs.shape))

    predicted_imgs = Wound.postprocessing(predicted_imgs)

    if predicted_imgs.shape[0] == 1:
        img = Image.fromarray(predicted_imgs[0], mode = 'L')
        img.save('predicted_img.jpeg')

        predicted_img_box, predicted_img_label = print_img_interface(img, 'predicted img', 1, 2)
    else:
        for i in range(predicted_imgs.shape[0]):
            img = Image.fromarray(predicted_imgs[i], mode = 'L')
            img.save('predicted_img_%d.jpeg' % i)

    logging.info('test time = %s' % (time.time() - t0))
    time_text.set('test time = %f' % (time.time() - t0))

def threshold_img_click():
    global truth_imgs, predicted_imgs, result_imgs
    global threshold_spin_box

    threshold = int(threshold_spin_box.get())
    if threshold > 255 or threshold < 0:
        logging.error('unexpected threshold = %d' % threshold)
        return

    # logging.info('threshold = %d' % threshold)

    result_imgs = np.zeros(predicted_imgs.shape, dtype = np.uint8)

    if predicted_imgs.shape[0] == 1:
        for y in range(predicted_imgs.shape[1]):
            for x in range(predicted_imgs.shape[2]):
                result_imgs[0][y][x] = 255 if predicted_imgs[0][y][x] > threshold else 0

        img = Image.fromarray(result_imgs[0], mode = 'L')
        img.save('result_img.jpeg')

        result_img_box, result_img_label = print_img_interface(img, 'result img', 3, 1)

        dc(truth_imgs[0], result_imgs[0])
    else:
        for i in range(predicted_imgs.shape[0]):
            for y in range(predicted_imgs.shape[1]):
                for x in range(predicted_imgs.shape[2]):
                    result_imgs[i][y][x] = 255 if predicted_imgs[i][y][x] > threshold else 0

def train_click():
    global time_text

    t0 = time.time()

    train()

    logging.info('training time = %s' % (time.time() - t0))
    time_text.set('training time = %f' % (time.time() - t0))

def predict_dir_click():
    global threshold_spin_box
    global time_text, dc_text

    test_wound = Wound(wound_test_original_dir, wound_test_truth_dir)

    t0 = time.time()

    predicted_imgs = predict(test_wound.ori_imgs, test_wound.truth_imgs)

    predicted_imgs = Wound.postprocessing(predicted_imgs)

    for i in range(predicted_imgs.shape[0]):
        img = Image.fromarray(predicted_imgs[i], mode = 'L')
        img.save(os.path.join(wound_result_dir, 'predicted_img_%d.jpeg' % i))

    logging.info('prediction time = %s' % (time.time() - t0))
    s = 'prediction time = %f\n' % (time.time() - t0)

    threshold = int(threshold_spin_box.get())
    if threshold > 255 or threshold < 0:
        logging.error('unexpected threshold = %d' % threshold)
        return

    result_imgs = np.zeros(predicted_imgs.shape, dtype = np.uint8)
    total_dc = 0.

    for i in range(predicted_imgs.shape[0]):
        for y in range(predicted_imgs.shape[1]):
            for x in range(predicted_imgs.shape[2]):
                result_imgs[i][y][x] = 255 if predicted_imgs[i][y][x] > threshold else 0
        
        img = Image.fromarray(result_imgs[i], mode = 'L')
        img.save(os.path.join(wound_result_dir, 'result_img_%d.jpeg' % i))

        total_dc += dc(test_wound.truth_imgs[i], result_imgs[i])
    
    total_dc /= predicted_imgs.shape[0]

    dc_text.set('mean dc value = %f' % total_dc)

    logging.info('predict dir time = %s' % (time.time() - t0))

    s += 'predict dir time = %f' % (time.time() - t0)

    time_text.set(s)


if __name__ == '__main__':
    # setup the tensorflow
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    win = tk.Tk()
    win.geometry('1200x1000')
    win.title('color segmentation')

    load_ori_img_but = tk.Button(win, text = 'load original img', command = load_ori_img_click)
    # load_ori_img_but.pack()
    load_ori_img_but.grid(row = 0, column = 0)

    load_truth_img_but = tk.Button(win, text = 'load ground truth img', command = load_truth_img_click)
    load_truth_img_but.grid(row = 0, column = 1)

    predict_but = tk.Button(win, text = 'predict img', command = predict_img_click)
    predict_but.grid(row = 0, column = 2)

    threshold_spin_box = tk.Spinbox(win, from_ = 200, to = 255)
    threshold_spin_box.grid(row = 0, column = 3)
    threshold_but = tk.Button(win, text = 'threshold img', command = threshold_img_click)
    threshold_but.grid(row = 1, column = 3)

    train_but = tk.Button(win, text = 'train', command = train_click)
    train_but.grid(row = 2, column = 3)

    predict_dir_but = tk.Button(win, text = 'predict whole dir imgs', command = predict_dir_click)
    predict_dir_but.grid(row = 3, column = 3)

    time_text = tk.StringVar()
    time_label = tk.Label(win, textvariable = time_text)
    time_label.grid(row = 5, column = 0)

    dc_text = tk.StringVar()
    dc_label = tk.Label(win, textvariable = dc_text)
    dc_label.grid(row = 6, column = 0)

    win.mainloop()

    # if len(sys.argv) >= 2:
    #     if sys.argv[1] == 'train':
    #         t0 = time.time()

    #         train()

    #         logging.info('training time = %s' % (time.time() - t0))
    #     elif sys.argv[1] == 'test' and len(sys.argv) >=3:
    #         t0 = time.time()

    #         # load original data
    #         ori_imgs = np.array([np.array(Image.open(sys.argv[2], 'r').convert('RGB'), dtype = np.float32)])

    #         # load ground truth data
    #         truth_imgs = np.array([np.array(Image.open(sys.argv[3], 'r').convert('L'), dtype = np.float32)])
    #         truth_imgs = np.expand_dims(truth_imgs, axis = len(truth_imgs.shape))
            
    #         # preprocess for truth_img so that the pixel range is from 0. to 1.
    #         truth_imgs /= 255.

    #         predicted_y = predict(ori_imgs, truth_imgs)

    #         # postprocess for truth_img so that the pixel range is from 0 to 255
    #         predicted_y = np.squeeze(predicted_y, axis = 3)
    #         predicted_y *= 255.
    #         predicted_y = predicted_y.astype(np.uint8)
            
    #         result_y = Image.fromarray(predicted_y[0], mode = 'L')
    #         result_y.save('result_y.jpeg')

    #         logging.info('test time = %s' % (time.time() - t0))
    # else:
    #     logging.error('unexpected sys.argv len = %d' % len(sys.argv))