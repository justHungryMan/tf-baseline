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
        self.initial_epoch = 0
        self.set_batch_size()

    def set_batch_size(self):
        self.batch_size = conf.hyperparameter[conf.base.target].batch_size
        self.conf.dataset.train.batch_size = self.conf.dataset.test.batch_size = self.batch_size
        
    def set_hyperparameter(self, conf, num_examples):
        lr = conf.hyperparameter[conf.base.target].learning_rate
        self.steps_per_epoch = (num_examples // self.batch_size)

        if conf.hyperparameter[conf.base.target].get('epoch', None):
            self.epoch = math.ceil(conf.hyperparameter[conf.base.target].steps / self.steps_per_epoch)
        else:
            self.epoch = conf.hyperparameter[conf.base.target].epoch

        # Upstream
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
        # Downstream
        elif conf.base.target == 'imagenet'
            boundaries = []
            values = []
            multistep_steps = [0.3, 0.6, 0.9]
            
            values.append(lr)

            for multistep_step in multistep_steps:
                boundaries.append(math.ceil(conf.hyperparameter[conf.base.target].steps * multistep_step))
                values.append(values[-1] * 0.1)
            logging.info(f'[Hyperparameter] steps: {self.steps_per_epoch * self.epoch} boundaries: {boundaries} values: {values}')
            conf.scheduler.steps = self.steps_per_epoch * self.epoch
            conf.scheduler.params.boundaries = boundaries
            conf.scheduler.params.values = values
        else:
            pass
    
    def build_optimizer(self):
        learning_rate_scheduler = baseline.scheduler.create(self.conf['scheduler'])
        params = {k: v for k, v in self.conf['optimizer']['params'].items()}
        params['learning_rate'] = learning_rate_scheduler
        opti_conf = {
            'type': self.conf['optimizer']['type'],
            'params': params
        }
        optimizer = baseline.optimizer.create(opti_conf)
        
        return optimizer
    
    def build_model(self, num_classes=1000):
        with self.strategy.scope():
            model = baseline.model.create(self.conf['model'], num_classes=num_classes)
            model.build((None, None, None, 3))
            if self.conf.base.get('pretrain'):
                self.load_weights(model)
                tf.keras.backend.set_value(model.optimizer.iterations, 0)
            self.load_chpt(model)

            optimizer = self.build_optimizer()
            loss_fn = baseline.loss.create(self.conf['loss'])

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'],
                                #steps_per_execution=1
                                )
            logging.info(f'Build Model Finish')
        model.summary()
        return model

    def load_weights(self, model):
        latest = tf.train.latest_checkpoint(self.conf.base.pretrain)
        model.load_weights(latest).expect_partial()
        logging.info(f'Model loaded from {self.conf.base.pretrain}')

    def load_chpt(self, model):
        if tf.io.gfile.isdir(self.conf.base.save_dir):
            if self.conf.base.resume is True:
                latest = tf.train.latest_checkpoint(self.conf.base.save_dir)
                model.load_weights(latest).expect_partial()
                self.initial_epoch = int(os.path.basename(latest).split('_')[-1])
                logging.info(f'Training resumed from {self.initial_epoch} epochs')
            else:
                tf.io.gfile.rmtree(self.conf.base.save_dir)
        else:
            tf.io.gfile.makedirs(self.conf.base.save_dir)
    
    def build_dataset(self):
        dataset = baseline.dataset.create(config=self.conf['dataset'], processing_config=self.conf['processing'], seed=self.conf.base.seed, num_devices=self.strategy.num_replicas_in_sync)
        train_dataset = self.strategy.experimental_distribute_dataset(dataset['train']['dataset'])
        test_dataset = self.strategy.experimental_distribute_dataset(dataset['test']['dataset'])

        return {
            "train": train_dataset, 
            "test": test_dataset,
            "train_info": dataset['train']['info'],
            "test_info": dataset['test']['info']
        }
    
    def build_callback(self):
        # TODO 
        for single_conf in self.conf.callbacks.modules:
            if single_conf['type'] == 'ModelCheckpoint':
                single_conf['params']['filepath'] = os.path.join(self.conf.base.save_dir, 'chpt_{epoch}')
            elif single_conf['type'] == 'TensorBoard':
                single_conf['params']['log_dir'] = f"{os.path.join(self.conf.base.save_dir, 'logs/fit/')}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = baseline.util.callback.create(self.conf.callbacks.modules)
        return callbacks

    def run(self):
        tf.keras.backend.clear_session()
        dataset = self.build_dataset()
        self.set_hyperparameter(conf=self.conf, num_examples=dataset['train_info']['num_examples'])
        model = self.build_model(num_classes=dataset['train_info']['num_classes'])
        callbacks = self.build_callback()

        mode = self.conf.base.mode

        if mode == 'train':
            self.train_eval(dataset=dataset, model=model, callbacks=callbacks)
        elif mode == 'train_eval':
            val_kwargs = {
                'validation_data': dataset['test'],
                'validation_steps': math.ceil(dataset['test_info']['num_examples'] / self.batch_size)
            } 
            self.train_eval(train_dataset=dataset['train'], model=model, callbacks=callbacks, val_kwargs=val_kwargs)
        elif mode == 'eval':
            pass
        elif mode == 'finetuning':
            
            pass
    
    def train_eval(self, train_dataset, model, callbacks, val_kwargs={}):
        logging.info(f'Start fit model')
        tic = time.time()
        history = model.fit(
            train_dataset,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epoch,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            verbose=1,
            **val_kwargs
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
    trainer.run()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)
    tf.get_logger().setLevel(logging.WARNING)
    main()