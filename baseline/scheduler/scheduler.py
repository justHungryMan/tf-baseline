from baseline.scheduler.warmup_lr import WarmUpExtension
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class WarmupPiecewiseConstantDecay(WarmUpExtension, PiecewiseConstantDecay):
    def __init__(self, warmup_step, init_lr, boundaries, values, name=None):
        super().__init__(warmup_step, init_lr, boundaries, values, name=name)