import tensorflow as tf

from .inception import center_crop, distort_crop, distort_color

def create(preprocess_list, info, decoder):
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
        image = data['image']

        label = data['label']
        
        label = tf.one_hot(tf.reshape(label, [-1]), info['num_classes'])
        label = tf.reduce_sum(label, axis=0)
        
        label = tf.clip_by_value(label, 0, 1)
        
        for preprocess_conf in preprocess_list:
            image, label = preprocess_image(preprocess_conf, image, label)

        return {'image': image, 'label': label}

    return _pp