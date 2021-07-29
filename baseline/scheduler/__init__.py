import logging
import tensorflow as tf
from . import scheduler

LOGGER = logging.getLogger(__name__)

def create(conf):
    
    if conf.type == 'warmup_piecewise':
        schduler = scheduler.WarmupPiecewiseConstantDecay(**conf['params'])
    else:
        raise AttributeError(f'not support scheduler config: {conf}')
    return schduler