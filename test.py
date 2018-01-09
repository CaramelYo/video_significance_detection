import sys
import logging

sys.stderr = open('stderr.txt', 'w', encoding = 'utf-8')
sys.stdout = open('stdout.txt', 'w', encoding = 'utf-8')

import tensorflow as tf
import numpy as np

# logging setting
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('test.log', 'w', 'utf-8'), ])

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)


# log_f_handler = logging.FileHandler('tensorflow.log', 'w', 'utf-8')
# log_f_handler.setLevel(logging.DEBUG)
# log_f_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s', datefmt = '%Y-%m-%d %H:%M'))
# logging.getLogger('tensorflow').addHandler(log_f_handler)
# logging.getLogger('tensorflow').setLevel(logging.DEBUG)


print('stdout')


# logging.debug('yooo')

# with tf.name_scope("foo"):
#     with tf.variable_scope("var_scope"):
#         v = tf.get_variable("var", [1])
# with tf.name_scope("bar"):
#     with tf.variable_scope("var_scope"):
#         v1 = tf.get_variable("var", [1])
# assert v1 == v
# print(v.name)   # var_scope/var:0
# print(v1.name)  # var_scope/var:0

t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
r = tf.pad(t, paddings, "CONSTANT")
# print(type(r))
# print(len(r))

rr = tf.slice(r, [1, 2], [2, 2])

p = tf.Print(rr, [rr],
            message = 'value of rr = ')

tf.logging.debug(rr)

# y = tf.py_func(logging.debug, [rr], tf.float32)
tf.logging.debug('yooo')

with tf.Session() as sess:
    r = sess.run(r)
    print(r)
    print('---------')
    # r = r[1: len(r) - 1]
    # r = r[:, 1:len(r[0]) - 1]
    # r = r[1: len(r) - 1, 1: len(r[0]) - 1]
    # print(r)
    rr = sess.run(rr)
    print(rr)

    tf.logging.debug(rr)

    print(sess.run(p))

    # sess.run(y)

    # print('yo' + sess.run(p))

# batch_size = 5

# l = [1, 2, 3]

# for i in range(1):
#     # print(i)
#     try:
#         ll = l[i * batch_size: (i + 1) * batch_size]
#     except Exception as e:
#         print(e)
    
#     print(ll)
#     print(len(ll))

# print(type(5 / 2))
# print(type(6 / 2))


t = np.asarray([[[1, 1], [2, 2], [3, 3]],
                [[4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9]],
                [[10,], [11, 11], [12, 12]]])

print(t)
print(t.shape)

print('----------------------')

# print(t.__attr)

# print(sys.stderr.__dict__)


raise('stderr')