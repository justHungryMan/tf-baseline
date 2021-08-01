import logging
import tensorflow as tf

from .MonitorCallback import MonitorCallback

LOGGER = logging.getLogger(__name__)

def create(conf):
    def create_callback(conf_callback):
        if conf_callback['type'] == 'MonitorCallback':
            return MonitorCallback()
        elif conf_callback['type'] == 'TerminateOnNaN':
            return tf.keras.callbacks.TerminateOnNaN()
        elif conf_callback['type'] == 'ProgbarLogger':
            return tf.keras.callbacks.ProgbarLogger(**conf_callback['params'])
        elif conf_callback['type'] == 'ModelCheckpoint':
            return tf.keras.callbacks.ModelCheckpoint(**conf_callback['params'])
        elif conf_callback['type'] == 'TensorBoard':
            return tf.keras.callbacks.TensorBoard(**conf_callback['params'])
        else:
            raise AttributeError(f'not support callback config: {conf_callback}')

    callbacks = []

    for single_conf in conf:
        callbacks.append(create_callback(single_conf))
    return callbacks