import logging
import tensorflow as tf

from .bce import BinaryCrossentropy

LOGGER = logging.getLogger(__name__)

def create(config):
    if config['type'].lower() == 'ce':
        LOGGER.info(f'[loss] create CategoricalCrossEntropy')

        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    elif config['type'].lower() == 'bce':
        LOGGER.info(f'[loss] create BinaryCrossEntropy')

        return BinaryCrossentropy()
    else:
        raise AttributeError(f'not support loss config: {config}')