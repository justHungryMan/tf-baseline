import sys
import os
import math
import copy
import random
import time
import logging
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf

import baseline

class Trainer():
    def __init__(self, conf):
        # super(Trainer, self).__init__(conf)
        self.conf = copy.deepcopy(conf)
        self.strategy = baseline.util.strategy.create(self.conf['base']['env'])
        self.dataset = baseline.dataset.create(config=self.conf['dataset'], seed=self.conf.base.seed, num_devices=self.strategy.num_replicas_in_sync)
        self.set_hyperparameter(conf=self.conf, num_examples=self.dataset['train']['info']['num_examples'])
        
    def set_hyperparameter(self, conf, num_examples):
        lr = conf.hyperparameter[conf.base.target].learning_rate
        self.batch_size = conf.hyperparameter[conf.base.target].batch_size
        self.epoch = conf.hyperparameter[conf.base.target].epoch
        self.steps_per_epoch = (num_examples // self.batch_size)

        if conf.base.target == 'bit-s':
            boundaries = []
            values = []
            multistep_epochs = [30, 60, 80]
            
            values.append(lr * self.batch_size / 256)
            
            for multistep_epoch in multistep_epochs:
                boundaries.append(self.steps_per_epoch * multistep_epoch)
                values.append(values[-1] * 0.1)
            
            logging.info(f'[Hyperparameter] steps: {self.steps_per_epoch * self.epoch} boundaries: {boundaries} values: {values}')
            conf.scheduler.steps = self.steps_per_epoch * self.epoch
            conf.scheduler.params.boundaries = boundaries
            conf.scheduler.params.values = values
    
    def build_optimizer(self, conf):
        learning_rate_scheduler = baseline.scheduler.create(conf['scheduler'])
        params = {k: v for k, v in conf['optimizer']['params'].items()}
        params['learning_rate'] = learning_rate_scheduler
        opti_conf = {
            'type': conf['optimizer']['type'],
            'params': params
        }
        optimizer = baseline.optimizer.create(opti_conf)
        
        return optimizer
    
        

    def train(self):
        pass
    
    def train_eval(self):
        with self.strategy.scope():
            self.model = baseline.model.create(self.conf['model'], num_classes=self.dataset['train']['info']['num_classes'])
            self.model.build((None, None, None, 3))
            optimizer = self.build_optimizer(self.conf)
            loss_fn = baseline.loss.create(self.conf['loss'])

            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics='accuracy',
                                steps_per_execution=1)
            logging.info(f'Build Model Finish')
        self.model.summary()
        train_dataset = self.strategy.experimental_distribute_dataset(self.dataset['train']['dataset'])
        test_dataset = self.strategy.experimental_distribute_dataset(self.dataset['test']['dataset'])

        validation_kwargs = {
            'validation_data': test_dataset,
            'validation_steps': math.ceil(self.dataset['test']['info']['num_examples'] / self.batch_size)
        } 

        logging.info(f'start fit model')
        tic = time.time()
        history = self.model.fit(
            train_dataset,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epoch,
            initial_epoch=0,
            callbacks=[tf.keras.callbacks.TerminateOnNaN(), baseline.util.callback.MonitorCallback(),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.conf.base.save_dir, 'chpt_{epoch}'),
                                            save_weights_only=True)],
            verbose=1,
            **validation_kwargs
        )
        toc = time.time()


    def eval(self):
        pass

def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        logging.info(f'[Seed] :{conf.base.seed}')

        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        tf.random.set_seed(conf.base.seed)




@hydra.main(config_path='conf', config_name='bit-s')
def main(conf : DictConfig) -> None:
    logging.info(f'Configuration\n{OmegaConf.to_yaml(conf)}')
    # Set Seed
    set_seed(conf)


    trainer = Trainer(conf)

    if conf.base.mode == 'train_eval':
        trainer.train_eval()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)
    tf.get_logger().setLevel(logging.WARNING)
    main()