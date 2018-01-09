import logging
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import math
import time

from data_preprocessing import Video
from FCN import FCN16


# class StreamToLogger(object):
#    """
#    Fake file-like stream object that redirects writes to a logger instance.
#    """
#    def __init__(self, logger, log_level=logging.INFO):
#       self.logger = logger
#       self.log_level = log_level
#       self.linebuf = ''

#    def write(self, buf):
#       for line in buf.rstrip().splitlines():
#          self.logger.log(self.log_level, line.rstrip())

#    def flush(self, buf):
#       self.write(buf)


# logging setting
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('main.log', 'w', 'utf-8'), ])

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

log_f_handler = logging.FileHandler('tensorflow.log', 'w', 'utf-8')
log_f_handler.setLevel(logging.DEBUG)
log_f_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s', datefmt = '%Y-%m-%d %H:%M'))
logging.getLogger('tensorflow').addHandler(log_f_handler)
logging.getLogger('tensorflow').setLevel(logging.DEBUG)


# log = logging.getLogger('test')
# log.setLevel(logging.DEBUG)
# # log(level = logging.DEBUG,
# #                     format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
# #                     datefmt = '%Y-%m-%d %H:%M',
# #                     handlers = [logging.FileHandler('main.log', 'w', 'utf-8'), ])

# console = logging.StreamHandler(sys.stdout)
# console.setLevel(logging.DEBUG)
# console.setFormatter(logging.Formatter('%(message)s'))
# log.addHandler(console)

# log_f_handler = logging.FileHandler('log.log', 'w', 'utf-8')
# log_f_handler.setLevel(logging.DEBUG)
# log_f_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s', datefmt = '%Y-%m-%d %H:%M'))
# log.addHandler(log_f_handler)

# print('yo std')
# log.debug('yooo')

# exit()


# stdout_logger = logging.getLogger('STDOUT')
# sl = StreamToLogger(stdout_logger, logging.INFO)
# sys.stdout = sl

# stderr_logger = logging.getLogger('STDERR')
# sl = StreamToLogger(stderr_logger, logging.ERROR)
# sys.stderr = sl

# print('yo')
# logging.debug('logging yo')

# exit()



# data
data_dir = 'data'
video_dir = os.path.join(data_dir, 'video')
vgg_path = os.path.join(data_dir, 'vgg16.npy')

# network setting

training_epochs = 2
display_epoch = 2
save_epoch = 3

batch_size = 2
display_batch = 5

# dropout, probability to keep units
dropout_rate = 0.75

model_dir = 'model'
model_name = 'fcn'
model_log_dir = 'model_log'


def get_total_batch(y):
    # # floor
    # return int(y.shape[0] / batch_size)
    # ceil
    return math.ceil(y.shape[0] / batch_size)
    # return math.ceil(y.get_shape()[0].value / batch_size)

def train():
    # load video data
    videos = []
    for name in os.listdir(video_dir):
        path = os.path.join(video_dir, name)
        if os.path.isdir(path):
            videos.append(Video(path))

    fcn = FCN16(vgg_path)
    optimizer, accuracy, loss = fcn.build(x, y, is_train = True, debug = True)

    # create a saver
    saver = tf.train.Saver()

    # start training
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        # for tensorboard
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(model_log_dir, graph = sess.graph)

        # encounter the size problem ???
        sess.run(init)

        # get total batches
        total_batches = []
        for i in range(len(videos)):
            total_batches.append(get_total_batch(videos[i].s_imgs))
            logging.debug('video %d has total_batch = %d' % (i, total_batches[i]))

        epoch = 0
        while epoch < training_epochs:
            for i in range(len(total_batches)):
                for j in range(total_batches[i]):
                    # get the batches of x and y
                    batch_x = videos[i].ori_imgs[j * batch_size: (j + 1) * batch_size]
                    batch_y = videos[i].s_imgs[j * batch_size: (j + 1) * batch_size]
                    # batch_y = batch_y.reshape([-1, 1])
                    # logging.debug('batch_y shape = %s' % str(batch_y.shape))

                    # # test whether data has ran out
                    # if batch_y.shape[0] != batch_size:
                    #     logging.error('data has run out')
                    #     # break
                        
                    _, loss_val = sess.run([optimizer, loss], feed_dict = {
                        x: batch_x,
                        y: batch_y
                    })

                    logging.debug('loss val in batch %d in video %d in epoch %d = %f' % (j, i, epoch, loss_val))

                    if j % display_batch == 0:
                        logging.debug('accuracy in batch %d in video %d in epoch %d = %f' % 
                                    (j, i, epoch, sess.run(accuracy, feed_dict = {
                                        x: batch_x,
                                        y: batch_y
                                    })))

                    summary_str = sess.run(merged_summary_op, feed_dict = {
                        x: batch_x,
                        y: batch_y
                    })

                    summary_writer.add_summary(summary_str, epoch * total_batches[i] + j)
                
            if epoch % save_epoch == 0:
                saver.save(sess, os.path.join(model_dir, model_name), global_step = epoch)
                logging.debug('save the model file %s in %s' % (model_name, model_dir))

            epoch += 1

    logging.debug('training has done')

def predict(data_x, data_y = None):
    fcn = FCN16(vgg_path)
    accuracy = fcn.build(x, y, debug = True)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        logging.debug('restoring model is completed')

        total_batch = get_total_batch(data_x)

        # get the prediction_tensor
        prediction_tensor = tf.get_collection('prediction_tensor')[0]

        # if data_y:
        #     # prediction and test the accuracy

        #     # accuracy = tf.get_collection('accuracy')[0]

        for i in range(total_batch):
            batch_x = data_x[i * batch_size: (i + 1) * batch_size]

            # test whether data_x has run out
            if batch_x.shape[0] != batch_size:
                # data has run out
                logging.error('data_x has run out')
            
            # predict
            predicted_y = sess.run(prediction_tensor, feed_dict = {
                x: batch_x
            })

            # logging.debug(type(data_y))

            if type(data_y) == np.ndarray:
                batch_y = data_y[i * batch_size: (i + 1) * batch_size]
                # batch_y = batch_y.reshape([-1, 1])

                logging.debug('accuracy in batch %d in = %f' % (i, sess.run(accuracy, feed_dict = {
                    x: batch_x,
                    y: batch_y
                })))

    logging.debug('prediction has been over')
    return predicted_y


if __name__ == '__main__':
    # setup the tensorflow
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32)
    # y ???
    y = tf.placeholder(tf.float32)

    # vars_dict = {
    #     ''
    # }

    # # show the histogram in tensorboard
    # for name in vars_dict.keys():
    #     tf.summary.histogram(name, vars_dict[name])

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            t0 = time.time()

            train()

            logging.debug('training time = %s' % (time.time() - t0))
        elif sys.argv[1] == 'predict' and len(sys.argv) >= 3:
            t0 = time.time()

            # load img data
            imgs = np.asarray([np.asarray(Image.open(sys.argv[2], 'r').convert('RGB'))])

            predicted_y = predict(imgs)
            # logging.debug(predicted_y)

            logging.debug('prediction time = %s' % (time.time() - t0))
        elif sys.argv[1] == 'test' and len(sys.argv) >=3:
            t0 = time.time()

            # load img data
            imgs = np.asarray([np.asarray(Image.open(sys.argv[2], 'r').convert('RGB'))])

            # load saliency data
            s_imgs = np.asarray([np.asarray(Image.open(sys.argv[3], 'r').convert('L'))])
            s_imgs = np.expand_dims(s_imgs, axis = len(s_imgs.shape))

            predicted_y = predict(imgs, s_imgs)
            # logging.debug(predicted_y)
            # logging.debug(type(predicted_y))
            # logging.debug(predicted_y.shape)
            # logging.debug(predicted_y[0].shape)
            predicted_y = predicted_y.reshape([1, 360, 640])
            # logging.debug(predicted_y.shape)
            with open('temp.txt', 'w', encoding = 'utf-8') as f:
                for y in range(len(predicted_y[0][0])):
                    for x in range(len(predicted_y[0])):
                        f.write('%f, ' % predicted_y[0][x][y])

                    f.write('\n')

            result_y = Image.fromarray(predicted_y[0], mode = 'L')
            # result_y.show()
            result_y.save('result_y.png')

            logging.debug('test time = %s' % (time.time() - t0))
    else:
        logging.error('unexpected sys.argv len = %d' % len(sys.argv))