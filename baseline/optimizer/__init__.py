import logging
import tensorflow as tf

LOGGER = logging.getLogger(__name__)

def create(conf):
    optimizer = tf.keras.optimizers.get({
        "class_name": conf['type'],
        "config": conf['params']
        })
        
    LOGGER.info(f'[Optimizer] create {conf["type"]}')
    return optimizer