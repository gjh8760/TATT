import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling
    
    Linearly increase in warm-up phase. Maximum value is d_model ^ (-.5) * n_warmup_steps ^ (-.5).
    After that, for epochs 1~7 use lr = n_steps ^ (.5).
    Use fixed lr = 1e-5 for epochs 8~10.
    '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_step=1):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.step = current_step
        self.init_lr = np.power(d_model, -0.5)
        self.phase2_epoch = 8
        self.phase2_lr = 0.00001

    def step_and_update_lr(self,epoch = 0):
        "Step with the inner optimizer"
        self._update_learning_rate(epoch)
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.step, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.step])

    def _update_learning_rate(self, epoch):
        ''' Learning rate scheduling per step '''
        lr = self.init_lr * self._get_lr_scale()
        if epoch >= self.phase2_epoch:
            lr = self.phase2_lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self.step += 1