from . import tfds
import tensorflow as tf

@tf.autograph.experimental.do_not_convert
def create(config, seed=None, num_devices=None):
    if config['train']['type'] == 'tensorflow_dataset':
        return {
            'train': tfds.create(config['train'], data_dir=config['data_dir'], seed=seed, num_devices=num_devices),
            'test': tfds.create(config['test'], data_dir=config['data_dir'], seed=seed, num_devices=num_devices)
        }
    else:
        raise AttributeError(f'not support dataset/type config: {config}')