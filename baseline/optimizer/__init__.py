import logging
import tensorflow as tf

LOGGER = logging.getLogger(__name__)

# def create(conf):
#     optimizer = tf.keras.optimizers.get({
#         "class_name": conf['type'],
#         "config": conf['params']
#         })
        
#     LOGGER.info(f'[Optimizer] create {conf["type"]}')
#     return optimizer

def create(config):
    opt_type = config['type'].lower()
    if opt_type == 'sgd':
        LOGGER.info(f'[optimizer] create {opt_type}')
        return tf.keras.optimizers.SGD(**config['params'])
    elif opt_type == 'sgdw':
        LOGGER.info(f'[optimizer] create {opt_type}')
        return tfa.optimizers.SGDW(**config['params'])
    elif opt_type == 'adam':
        LOGGER.info(f'[optimizer] create {opt_type}')
        return tf.keras.optimizers.Adam()
    elif opt_type == 'adamw':
        LOGGER.info(f'[optimizer] create {opt_type}')
        return tfa.optimizers.AdamW(**config['params'])
    elif opt_type == 'lamb':
        LOGGER.info(f'[optimizer] create {opt_type}')
        return tfa.optimizers.LAMB(**config['params'])

    raise AttributeError(f'not support optimizer config: {config}')