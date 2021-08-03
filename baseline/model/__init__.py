import logging
import tensorflow as tf

from . import bit

LOGGER = logging.getLogger(__name__)


def create(conf, num_classes=1000):
    base, architecture_name = [l.lower() for l in conf['type'].split('/')]

    if base == 'bit':
        model = bit.create_name(conf['type'].split('/')[-1], num_outputs=num_classes, weight_decay=conf['weight_decay'])
        LOGGER.info(f'[Model] create {architecture_name}')
    else:
        raise AttributeError(f'not support architecture config: {conf}')

    return model