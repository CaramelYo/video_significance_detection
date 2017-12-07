import logging
import sys
import os
import tensorflow as tf


# logging setting
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('main.log', 'w', 'utf-8'), ])

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    logging.debug('hello')
