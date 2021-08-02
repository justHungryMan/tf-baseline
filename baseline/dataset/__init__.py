from . import tfds
import tensorflow as tf

def create(config, processing_config, seed=None, num_devices=None):
    if config['type'] == 'tensorflow_dataset':
        train_config = config['train']
        train_config.update(processing_config['train'])

        test_config = config['test']
        test_config.update(processing_config['test'])

        return {
            'train': tfds.create(train_config, data_dir=config['data_dir'], seed=seed, num_devices=num_devices),
            'test': tfds.create(test_config, data_dir=config['data_dir'], seed=seed, num_devices=num_devices)
        }
    else:
        raise AttributeError(f'not support dataset/type config: {config}')