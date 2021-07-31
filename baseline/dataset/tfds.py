import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import os
import json

from .inception import center_crop, distort_crop, distort_color


# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000


def create(config, data_dir=None, seed=None, num_devices=1):
    mode = 'label'
    
    builder = tfds.builder(config['name']) if data_dir is None else tfds.builder(config['name'], data_dir=data_dir)
    builder.download_and_prepare(
        # download_config=tfds.download.DownloadConfig(manula_dir='~/tensorflow_datasets/')
    )
    info = {
        'num_examples': builder.info.splits[config['split']].num_examples,
        'num_shards': len(builder.info.splits[config['split']].file_instructions),
        'num_classes': get_num_classes(config['name'], config.get('task', None), builder) if mode == 'label' else config['num_classes']
    }

    dataset = builder.as_dataset(split=config['split'],
                                shuffle_files=config.get('shuffle', False),
                                decoders={'image': tfds.decode.SkipDecoding()})
    decoder = builder.info.features['image'].decode_example

    examples_per_class = config.get('examples_per_class', -1)
    assert examples_per_class == -1

    dataset = dataset.cache()

    if config.get('repeat', False):
        dataset = dataset.repeat()

    if config.get('shuffle', False):
        dataset = dataset.shuffle(min(info['num_examples'], MAX_IN_MEMORY))

    dataset = dataset.map(preprocess(config, config['preprocess'], info, decoder, mode), tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(config['batch_size'], drop_remainder=config['drop_remainder'])
    dataset = postprocess(config.get('postprocess', []), dataset)
    dataset = dataset.map(lambda v: (v['image'], v['label']))
    
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # from deepmind's code : https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L91
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    policy = tf.data.experimental.AutoShardPolicy.FILE if info['num_shards'] > num_devices else tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_distribute.auto_shard_policy = policy
    dataset = dataset.with_options(options)
    
    return {'dataset': dataset, 'info': info}

def preprocess(config, preprocess_list, info, decoder, mode):
    def preprocess_image(conf, image, label):
        if conf['type'] == 'resize':
            return tf.image.resize(image, **{
                'size': [224, 224],
                'method': 'bicubic'
            }), label
        elif conf['type'] == 'random_crop':
            return tf.image.random_crop(image, **conf['params']), label
        elif conf['type'] == 'random_flip':
            return tf.image.random_flip_left_right(image), label
        elif conf['type'] == 'normalize':
            return (image - conf['params']['mean']) / conf['params']['std'], label
        elif conf['type'] == 'imagenet_normalize':
            MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            return (image - MEAN_RGB) / STDDEV_RGB, label
        elif conf['type'] in {'labelsmooth', 'label_smooth', 'label_smoothing'}:
            return image, label * (1.0 - conf['params']['epsilon']) / (conf['params']['epsilon'] / info['num_classes'])
        elif conf['type'] in {'inception_random_crop'}:
            return distort_crop(image, **conf['params']), label
        elif conf['type'] in {'inception_center_crop'}:
            return center_crop(image, **conf['params']), label
        elif conf['type'] in {'inception_distort_color'}:
            return distort_color(image, **conf.get('params', {})), label
        elif conf['type'] in {'cast'}:
            dtype = tf.bfloat16 if conf['params']['type'] == 'bfloat16' else None
            dtype = tf.float32 if conf['params']['type'] == 'float32' else dtype
            dtype = tf.float16 if conf['params']['type'] == 'float16' else dtype
            dtype = tf.uint8 if conf['params']['type'] == 'uint8' else dtype
            dtype = tf.int32 if conf['params']['type'] == 'int32' else dtype
            if dtype is None:
                raise AttributeError(f'not support cast type: {conf}')
            return tf.image.convert_image_dtype(image, dtype=dtype), label       
        else:
            raise AttributeError(f'not support dataset/preprocess config: {conf}')
    
    def _pp(data):
        data['image'] = decoder(data['image'])
        data = task_preprocessing(config['task'])(data) if config.get('task', None) is not None else data
        image = data['image']

        label = data[mode]
        
        label = tf.one_hot(tf.reshape(label, [-1]), info['num_classes'])
        label = tf.reduce_sum(label, axis=0)
        
        label = tf.clip_by_value(label, 0, 1)
        
        for preprocess_conf in preprocess_list:
            image, label = preprocess_image(preprocess_conf, image, label)

        return {'image': image, 'label': label}

    return _pp


def postprocess(config, dataset):
    def create_process(conf):
        if conf['type'] == 'mixup':
            alpha = conf['params']['alpha']

            def _mixup(data):
                beta_dist = tfp.distributions.Beta(alpha, alpha)
                beta = tf.cast(beta_dist.sample([]), tf.float32)
                data['image'] = (beta * data['image'] + (1 - beta) * tf.reverse(data['image'], axis=[0]))
                data['label'] = (beta * data['label'] + (1 - beta) * tf.reverse(data['label'], axis=[0]))
                return data
            return _mixup
        else:
            raise AttributeError(f'not support dataset/postprocess config: {conf}')

    for conf in config:
        dataset = dataset.map(create_process(conf), tf.data.experimental.AUTOTUNE)
    
    return dataset


# special for VTAB implements

def get_num_classes(name, task, builder):
    label_name = 'label'

    if 'dsprites' in name:
        if task == 'orientation':
            label_name = f'label_{task}'
        elif task == 'location':
            label_name = 'label_x_position'
    elif 'smallnorb' in name:
        label_name = f'label_{task}'

    if 'clevr' in name:
        if task == 'count':
            num_classes = 8
        elif task == 'distance':
            num_classes = 6
    elif 'kitti' in name:
        num_classes = 4
    elif label_name in builder.info.features:
        num_classes = builder.info.features[label_name].num_classes
    else:
        # pos & negative class
        # builder.info.features['pos'].num_classes == builder.info.features['neg'].num_classes
        num_classes = builder.info.features['pos'].num_classes

    return num_classes


def task_preprocessing(task):
    # clevr -->
    def _count_preprocess_fn(x):
        return {
            'image': x["image"], 
            'label': tf.size(x["objects"]["size"]) - 3
        }


    def _count_cylinders_preprocess_fn(x):
        # Class distribution:

        num_cylinders = tf.reduce_sum(
            tf.cast(tf.equal(x["objects"]["shape"], 2), tf.int32))
        return (x["image"], num_cylinders)


    def _closest_object_preprocess_fn(x):
        dist = tf.reduce_min(x["objects"]["pixel_coords"][:, 2])
        # These thresholds are uniformly spaced and result in more or less balanced
        # distribution of classes, see the resulting histogram:

        thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
        label = tf.reduce_max(tf.where((thrs - dist) < 0))
        return {
            'image': x["image"], 
            'label': label
        }
    # --> clevr

    # dsprites -->
    def _dsprites_location_preprocess_fn(x):
        # For consistency with other datasets, image needs to have three channels
        # and be in [0, 255).
        image = x['image']
        image = tf.tile(image, [1, 1, 3]) * 255
        label = tf.cast(
            tf.math.floordiv(
                tf.cast(x["label_x_position"], tf.float32),
                2.), tf.int64)
        return {
            'image': image, 
            'label': label
        }

    def _dsprites_orientation_preprocess_fn(x):
        
        # For consistency with other datasets, image needs to have three channels
        # and be in [0, 255).
        image = x['image']
        image = tf.tile(image, [1, 1, 3]) * 255
        label = x["label_orientation"]
        return {
            'image': image, 
            'label': label
        }
    # --> dsprites

    # smallnorb -->
    def _smallnorb_azimuth_preprocess_fn(x):
        # For consistency with other datasets, image needs to have three channels
        # and be in [0, 255).
        image = x['image']
        image = tf.tile(image, [1, 1, 3]) * 255

        label = x["label_azimuth"]
        return {
            'image': image, 
            'label': label
        }

    def _smallnorb_elevation_preprocess_fn(x):
        # For consistency with other datasets, image needs to have three channels
        # and be in [0, 255).
        image = x['image']
        image = tf.tile(image, [1, 1, 3]) * 255

        label = x["label_elevation"]
        return {
            'image': image, 
            'label': label
        }
    # --> smallnorb

    # kitti -->
    def _kitti_count_vehicles_preprocess_fn(x):
        """Counting vehicles."""
        # Label distribution:
        vehicles = tf.where(x["objects"]["type"] < 3)  # Car, Van, Truck.
        # Cap at 3.
        label = tf.math.minimum(tf.size(vehicles), 3)
        return {
            'image': x["image"], 
            'label': label
        }

    # --> kitti
    if task == 'count':
        return _count_preprocess_fn
    elif task == 'distance':
        return _closest_object_preprocess_fn
    elif task == 'location':
        return _dsprites_location_preprocess_fn
    elif task == 'orientation':
        return _dsprites_orientation_preprocess_fn
    elif task == 'azimuth':
        return _smallnorb_azimuth_preprocess_fn
    elif task == 'elevation':
        return _smallnorb_elevation_preprocess_fn
    elif task == 'kitti_count':
        return _kitti_count_vehicles_preprocess_fn