import os
import torch
from torch import optim
import torch.distributed as dist
import math
import time
import datetime
from statistics import mean
from cdistnet.optim.loss import cal_performance
from cdistnet.optim.optim import ScheduledOptim

class Trainer:
    def __init__(
                self,
                model,
                saved_model,
                train_dataloader,
                val_dataloaders,
                num_epochs,
                logger,
                tb_logger,
                display_iter,
                tfboard_iter,
                val_iter,
                model_dir,
                label_smoothing,
                grads_clip,
                cfg,
                args
                ):
        self.model = model
        self.saved_model = saved_model
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders
        self.num_epochs = num_epochs
        self.logger = logger
        self.tb_logger = tb_logger
        self.display_iter = display_iter
        self.tfboard_iter = tfboard_iter
        self.val_iter = val_iter
        self.model_dir = model_dir
        self.label_smoothing = label_smoothing
        self.grads_clip = grads_clip
        self.cfg = cfg
        self.args = args
        self.is_master = cfg.dist_train == False or cfg.local_rank == 0
        self.dist_train = cfg.dist_train

        self.current_step = 1
        self.start_epoch = 1
        self.BEST_val_acc = 0.
        self.BEST_epoch = 0
        self.BEST_step = 0
        self.BEST_global_step = 0
        self.set_device()
        if self.saved_model:
            self.load_checkpoint()
        self.model_to_device()
        self.configure_optimizers()

    def set_device(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if self.cfg.dist_train:
            torch.cuda.set_device(self.args.local_rank)
            self.device = self.args.local_rank
        else:
            torch.cuda.set_device(f'cuda:{self.args.gpu}')
            self.device = torch.device('cuda')

    def load_checkpoint(self): 
        ckpt = torch.load(self.saved_model)
        self.model.load_state_dict(ckpt['model'])
        self.load_epoch = ckpt['epoch']
        self.start_epoch = self.load_epoch + 1
        self.BEST_val_acc = ckpt['BEST_val_acc']
        self.BEST_epoch = ckpt['BEST_epoch']
        self.BEST_step = ckpt['BEST_step']
        self.BEST_global_step = ckpt['BEST_global_step']
        self.current_step = self.current_epoch * len(self.train_dataloader) + 1

    def model_to_device(self):
        if self.cfg.dist_train:
            # distributed training
            self.model.cuda(self.args.local_rank)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.model = self.model.to(self.device)

    def configure_optimizers(self):
        self.optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            betas=(0.9, 0.98),
            eps=1e-09,
        ),
        self.cfg.hidden_units, self.cfg.train.learning_rate_warmup_steps, self.current_step)

    def save_checkpoint(self, ckpt_name):
        ckpt = dict()
        ckpt['model'] = self.model.state_dict() if not self.dist_train else self.model.module.state_dict()
        ckpt['epoch'] = self.epoch
        ckpt['BEST_val_acc'] = self.BEST_val_acc
        ckpt['BEST_epoch'] = self.BEST_epoch
        ckpt['BEST_step'] = self.BEST_step
        ckpt['BEST_global_step'] = self.BEST_global_step
        torch.save(ckpt, f'{self.model_dir}/{ckpt_name}')

    def fit(self):
        for self.epoch in range(self.start_epoch, self.num_epochs + 1):
            if self.cfg.dist_train:
                self.train_dataloader.sampler.set_epoch(self.epoch)
                for val_dataloader in self.val_dataloaders:
                    val_dataloader.sampler.set_epoch(self.epoch)
            start = time.time()
            train_loss, train_char_acc, train_word_acc = self.training_step()
            if self.is_master:
                self.logger.info(f'  - (Trained epoch)   loss: {train_loss: 8.5f}, \
                                    char acc: {train_char_acc:3.3f}, word acc: {train_word_acc:3.3f}, \
                                    time: {(time.time() - start) / 60:3.3f} min')
        
    def training_step(self):
        self.model.train()
        total_loss = 0
        n_char_total = 0
        n_correct_char_total = 0
        n_word_total = 0
        n_correct_word_total = 0

        max_steps = len(self.train_dataloader)
        end = time.time()
        total_time = 0.
        count = 0

        dl_start = time.time()
        for step, batch in enumerate(self.train_dataloader):
            global_step = max_steps * (self.epoch - 1) + step
            if self.args.initial_test and step == 0 and self.epoch == 1:
                # test model save and validation
                if self.is_master:
                    self.logger.info("Testing model save")
                    self.save_checkpoint('save_test.pth')
                    os.remove(os.path.join(self.model_dir, 'save_test.pth'))
                    self.logger.info("Pass")
                    self.logger.info("Testing validation")
                    val_loss, val_char_acc, val_word_acc, val_word_accs = self.validation_step()
                    self.logger.info("Pass")
            dl_end = time.time()
            # print(f'dl: {dl_end - dl_start}') # data loading time
            dt_start = time.time()
            if self.is_master:
                self.tb_logger.update_step(global_step)
            if self.cfg.dist_train:
                images = batch[0].cuda(self.device, non_blocking=True)
                tgt = batch[1].cuda(self.device, non_blocking=True)
            else:
                images = batch[0].to(self.device)
                tgt = batch[1].to(self.device)
            dt_end = time.time()
            # print(f'dt : {dt_end - dt_start}') # data transfer time
            ff_start = time.time()

            pred = self.model(images, tgt) # (B T) K
            tgt = tgt[:, 1:]
            loss, n_correct_char, n_char, n_correct_word, n_word = cal_performance(pred, tgt, smoothing=self.label_smoothing) # TODO: why loss nan?
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grads_clip)
            self.optimizer.step_and_update_lr(self.epoch)

            total_loss += loss.item()
            n_char_total += n_char
            n_correct_char_total += n_correct_char
            n_word_total += n_word
            n_correct_word_total += n_correct_word
            
            batch_time = time.time() - end
            end = time.time()
            total_time += batch_time
            count += 1
            avg_time = total_time / count
            eta_seconds = avg_time * (max_steps - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            char_acc = n_correct_char_total / n_char_total
            word_acc = n_correct_word_total / n_word_total

            ff_end = time.time()
            # print(f'ff : {ff_end - ff_start}') # feedforward time
            
            if self.is_master:
                if self.tfboard_iter and (max_steps * (self.epoch - 1) + step) % self.tfboard_iter == 0:
                    self.tb_logger.add_scalar(
                        lr=self.optimizer._optimizer.param_groups[0]['lr'],
                        train_loss=loss.item(),
                        train_acc=word_acc)

                if step % self.display_iter == 0:
                    msg = 'epoch: {epoch}  iter: {iter}  loss: {loss: .6f}  lr: {lr: .6f}  eta: {eta}'.format(
                            epoch=self.epoch,
                            iter='{}/{}'.format(step, max_steps),
                            loss=loss.item(),
                            lr=self.optimizer._optimizer.param_groups[0]['lr'],
                            eta=eta_string
                        )
                    self.logger.info(msg)

                if self.epoch >= self.cfg.train.val_start_epoch and step % self.val_iter == 0:
                    val_loss, val_char_acc, val_word_acc, val_word_accs = self.validation_step()
                    self.tb_logger.add_scalar(
                        val_loss=val_loss,
                        val_acc=val_word_acc)
                    self.logger.info(f'val_loss:{val_loss:.4f}, val_char_acc:{val_char_acc:.4f}, val_word_acc:{val_word_acc:.4f}--------\n')
                    if val_word_acc > self.BEST_val_acc:
                        self.BEST_val_acc = val_word_acc
                        self.BEST_epoch = self.epoch
                        self.BEST_step = step
                        self.BEST_global_step = max_steps * (self.epoch - 1) + step
                        self.logger.info(f"Saving model: best_acc in epoch:{self.BEST_epoch}, step:{self.BEST_step}, val_word_acc:{self.BEST_val_acc:.4f}")
                        self.save_checkpoint(f'epoch{self.epoch:02d}_best_acc.pth')
                        self.logger.info("Saved!\n")
                    self.model.train()

            dl_start = time.time()

        b_loss = total_loss / max_steps
        char_acc = n_correct_char_total / n_char_total
        word_acc = n_correct_word_total / n_word_total
        if self.is_master:
            self.logger.info(f"Now: best_word_acc in epoch:{self.BEST_epoch}, step:{self.BEST_step}, val_word_acc: {self.BEST_val_acc}")
        return b_loss, char_acc, word_acc


    def validation_step(self):
        self.model.eval()
        
        val_losses = []
        val_char_accs = []
        val_word_accs = []

        with torch.no_grad():
            for val_dataloader in self.val_dataloaders:
                total_loss = 0
                n_char_total = 0
                n_correct_char_total = 0
                n_word_total = 0
                n_correct_word_total = 0

                for batch in val_dataloader:
                    if self.cfg.dist_train:
                        images = batch[0].cuda(self.device, non_blocking=True)
                        tgt = batch[1].cuda(self.device, non_blocking=True)
                    else:
                        images = batch[0].to(self.device)
                        tgt = batch[1].to(self.device)
                    pred = self.model(images, tgt)
                    tgt = tgt[:, 1:]
                    loss, n_correct_char, n_char, n_correct_word, n_word = cal_performance(pred, tgt, smoothing=self.label_smoothing)
                    
                    total_loss += loss.item()
                    n_char_total += n_char
                    n_correct_char_total += n_correct_char
                    n_word_total += n_word
                    n_correct_word_total += n_correct_word

                val_loss = total_loss / len(val_dataloader)
                val_losses.append(val_loss)
                val_char_acc = n_correct_char_total / n_char_total
                val_char_accs.append(val_char_acc)
                val_word_acc = n_correct_word_total / n_word_total
                val_word_accs.append(val_word_acc)

        return mean(val_losses), mean(val_char_accs), mean(val_word_accs), val_word_accs


