import sys
import os
import copy
from copy import deepcopy
from IPython import embed
import time
from datetime import datetime
from tqdm import tqdm
import math
import random

import numpy as np
import cv2
import imageio
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from interfaces import base

from utils import util, ssim_psnr
from utils import utils_moran
from utils.meters import AverageMeter
from utils.metrics import get_string_aster, get_string_crnn, Accuracy
from utils.util import str_filt

from model import gumbel_softmax
from loss.semantic_loss import SemanticLoss

from tensorboardX import SummaryWriter

from ptflops import get_model_complexity_info

import editdistance

import lpips


sys.path.append('../')
sys.path.append('./')

lpips_vgg = lpips.LPIPS(net="vgg")

sem_loss = SemanticLoss()
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')

ssim = ssim_psnr.SSIM()
tri_ssim = ssim_psnr.TRI_SSIM()


class TextSR(base.TextBase):

    def loss_stablizing(self, loss_set, keep_proportion=0.7):

        # acsending
        sorted_val, sorted_ind = torch.sort(loss_set)
        batch_size = loss_set.shape[0]

        # print("batch_size:", loss_set, batch_size)
        loss_set[sorted_ind[int(keep_proportion * batch_size)]:] = 0.0

        return loss_set

    def torch_rotate_img(self, torch_image_batches, arc_batches, rand_offs, off_range=0.2):

        # ratios: H / W
        device = torch_image_batches.device

        N, C, H, W = torch_image_batches.shape
        ratios = H / float(W)

        # rand_offs = random.random() * (1 - ratios)
        ratios_mul = ratios + (rand_offs.unsqueeze(1) * off_range * 2) - off_range  # [-0.05, 0.45]

        a11 = torch.cos(arc_batches)
        a12 = torch.sin(arc_batches)
        a21 = -torch.sin(arc_batches)
        a22 = torch.cos(arc_batches)

        # print("rand_offs:", rand_offs.shape, a12.shape)

        x_shift = torch.zeros_like(arc_batches)
        y_shift = torch.zeros_like(arc_batches)

        # print("device:", device)
        affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1) * ratios_mul, x_shift.unsqueeze(1),
                                   a21.unsqueeze(1) / ratios_mul, a22.unsqueeze(1), y_shift.unsqueeze(1)], dim=1)
        affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

        affine_grid = F.affine_grid(affine_matrix, torch_image_batches.shape)
        distorted_batches = F.grid_sample(torch_image_batches, affine_grid)

        return distorted_batches

    def model_inference(self, images_lr, images_hr, model_list, aster, i):
        ret_dict = {}   # keys: label_vecs, duration, pr_weights, images_sr
        ret_dict["label_vecs"] = None
        ret_dict["duration"] = 0
        ###############################################################################################################
        ## 원래 코드 시작 ##
        # cascade_images = images_lr
        #
        # images_sr = []
        #
        # for m_iter in range(self.args.stu_iter):
        #     if self.args.tpg_share:
        #         tpg_pick = 0
        #     else:
        #         tpg_pick = m_iter
        #
        #     stu_model = aster[1][tpg_pick]
        #     aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :])  # cascade_images
        #     before = time.time()
        #     label_vecs_logits = stu_model(aster_dict_lr)
        #     after = time.time()
        #
        #     ret_dict["duration"] += (after - before)
        #
        #     label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
        #     label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        #     ret_dict["label_vecs"] = label_vecs
        #
        #     # image for cascading
        #     if self.args.for_cascading:
        #         if i > 0:
        #             cascade_images = nn.functional.interpolate(cascade_images,
        #                                                        (
        #                                                            self.config.TRAIN.height // self.scale_factor,
        #                                                            self.config.TRAIN.width // self.scale_factor),
        #                                                        mode='bicubic')
        #             # cascade_images = model(cascade_images, label_vecs_final)
        #             cascade_images[cascade_images > 1.0] = 1.0
        #             cascade_images[cascade_images < 0.0] = 0.0
        #
        #             cascade_images = (cascade_images + images_lr) / 2
        #
        #     if self.args.sr_share:
        #         pick = 0
        #     else:
        #         pick = m_iter
        #
        #     before = time.time()
        #     cascade_images, pr_weights = model_list[pick](images_lr,
        #                                                   label_vecs_final.detach())
        #     after = time.time()
        #
        #     ret_dict["pr_weights"] = pr_weights
        #
        #     # print("fps:", (after - before))
        #     ret_dict["duration"] += (after - before)
        #
        #     images_sr.append(cascade_images)
        #
        # channel_num = 4
        #
        # before = time.time()
        # images_sr = model_list[0](images_lr[:, :channel_num, ...])  # ??????
        # after = time.time()
        #
        # ret_dict["duration"] += (after - before)
        # ret_dict["images_sr"] = images_sr
        ## 원래 코드 끝 ##
        ###############################################################################################################

        ###############################################################################################################
        ## 수정 코드 시작 ##
        stu_model = aster[1][0]
        aster_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])  # cascade_images
        before = time.time()
        label_vecs_logits = stu_model(aster_dict_lr)
        after = time.time()

        ret_dict["duration"] += (after - before)    # TP 뽑는 데 걸리는 시간

        label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
        label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        ret_dict["label_vecs"] = label_vecs

        before = time.time()
        images_sr, pr_weights = model_list[0](images_lr, label_vecs_final.detach())
        after = time.time()

        ret_dict["pr_weights"] = pr_weights

        # print("fps:", (after - before))
        ret_dict["duration"] += (after - before)    # SR 이미지 뽑는 데 걸리는 시간
        ret_dict["images_sr"] = images_sr
        ## 수정 코드 끝 ##
        ###############################################################################################################
        return ret_dict

    def train(self):
        _DEBUG = False
        # TP_Generator_dict = {"CRNN": self.CRNN_init, "OPT": self.TPG_init}
        TP_Generator_dict = {'CRNN': self.CRNN_init}

        tpg_opt = self.opt_TPG

        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']

        model_list = [model]
        if not self.args.sr_share:
            for i in range(self.args.stu_iter - 1):
                model_sep = self.generator_init(i+1)['model']
                model_list.append(model_sep)
        # else:
        #     model_list = [model]

        tensorboard_dir = os.path.join("tensorboard", self.vis_dir)
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        else:
            print("Directory exist, remove events...")
            os.popen("rm " + tensorboard_dir + "/*")

        self.results_recorder = SummaryWriter(tensorboard_dir)

        # aster: 학습 X, pretrain 된 crnn 사용
        aster, aster_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt)

        test_bible = {}

        if self.args.test_model == "CRNN":
            crnn, aster_info = self.TPG_init(recognizer_path=None, opt=tpg_opt) if self.args.CHNSR else self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_OPT_data if self.args.CHNSR else self.parse_crnn_data,
                        'string_process': get_string_crnn
                    }

        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }

        # print("self.args.arch:", self.args.arch)
        aster_student = []
        stu_iter = self.args.stu_iter

        for i in range(stu_iter):
            recognizer_path = os.path.join(self.resume, "recognizer_best.pth")
            # print("recognizer_path:", recognizer_path)
            if os.path.isfile(recognizer_path):
                # aster_student:
                aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=recognizer_path, opt=tpg_opt)
            else:
                aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt)

            if type(aster_student_) == list:
                aster_student_ = aster_student_[i]

            aster_student_.train()
            aster_student.append(aster_student_)

        aster.eval()
        if self.args.use_label:
            aster.train()

        # Recognizer needs to be fixed:
        # aster
        optimizer_G = self.optimizer_init(model_list)
        # for p in aster.parameters():
        #     p.requires_grad = False

        #print("cfg:", cfg.ckpt_dir)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []
        lr = cfg.lr

        for model in model_list:
            model.train()

        for epoch in range(cfg.epochs):

            for j, data in (enumerate(train_loader)):
                iters = len(train_loader) * epoch + j + 1

                if not self.args.go_test:
                    for model in model_list:
                        for p in model.parameters():
                            p.requires_grad = True
                    images_hrraw, images_pseudoLR, images_lrraw, images_HRy, images_lry, label_strs, label_vecs, weighted_mask, weighted_tics = data
                    text_label = label_vecs

                    images_lr = images_lrraw.to(self.device)
                    images_hr = images_hrraw.to(self.device)

                    if self.args.rotate_train:
                        # print("We are in rotate_train", self.args.rotate_train)
                        batch_size = images_lr.shape[0]

                        angle_batch = np.random.rand(batch_size) * self.args.rotate_train * 2 - self.args.rotate_train
                        arc = angle_batch / 180. * math.pi
                        rand_offs = torch.tensor(np.random.rand(batch_size)).float()

                        arc = torch.tensor(arc).float()
                        images_lr_ret = images_lr.clone()
                        images_hr_ret = images_hr.clone()

                        images_lr = self.torch_rotate_img(images_lr, arc, rand_offs)
                        images_hr = self.torch_rotate_img(images_hr, arc, rand_offs)


                        # print(images_lr.shape, images_hr.shape)

                    # print("iters:", iters)

                    loss_tssim = torch.tensor(0.)

                    aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :])
                    label_vecs_logits_hr = aster(aster_dict_hr)
                    label_vecs_hr = torch.nn.functional.softmax(label_vecs_logits_hr, -1).detach()

                    cascade_images = images_lr

                    loss_img = 0.
                    loss_recog_distill = 0.

                    cascade_images = cascade_images.detach()

                    tpg_pick = 0
                    # stu_model: extract prob. vec. from interpolated LR image, trainable
                    stu_model = aster_student[tpg_pick]

                    aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :])

                    label_vecs_logits = stu_model(aster_dict_lr)
                    label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)

                    # label_vecs_final: TP from LR image
                    label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                    # image for cascading
                    pick = 0
                    cascade_images, ret_mid = model_list[pick](images_lr, label_vecs_final.detach())

                    # [N, C, H, W] -> [N, T, C]
                    # text_label = text_label.squeeze(2).permute(2, 0, 1)
                    if self.args.use_distill:
                        loss_recog_distill_each = sem_loss(label_vecs, label_vecs_hr) * 100 #100
                        loss_recog_distill += loss_recog_distill_each  # * (1 + 0.5 * i)

                    im_quality_loss = image_crit(cascade_images, images_hr)

                    if self.args.training_stablize:
                        im_quality_loss = self.loss_stablizing(im_quality_loss)

                    loss_img_each = im_quality_loss.mean() * 100

                    # loss_img += loss_img_each * (1 + i * 0.5)
                    loss_img += loss_img_each

                    if self.args.ssim_loss:
                        loss_ssim = (1 - ssim(cascade_images, images_hr).mean()) * 10.
                        loss_img += loss_ssim

                    if self.args.tssim_loss:

                        cascade_images_sr_ret, srret_mid = model_list[pick](images_lr_ret, label_vecs_final.detach())
                        cascade_images_ret = self.torch_rotate_img(cascade_images_sr_ret, arc, rand_offs)
                        loss_tssim = (1 - tri_ssim(cascade_images_ret, cascade_images, images_hr).mean()) * 10.
                        loss_img += loss_tssim

                    if iters % 5 == 0 and i == self.args.stu_iter - 1:

                        # self.results_recorder.add_scalar('loss/distill', float(loss_recog_distill_each.data) * 100,
                        #                                  global_step=iters)
                        #
                        # self.results_recorder.add_scalar('loss/SR', float(loss_img_each.data) * 100,
                        #                                  global_step=iters)
                        #
                        # self.results_recorder.add_scalar('loss/SSIM', float(loss_ssim) * 100,
                        #                                  global_step=iters)

                        self.results_recorder.add_scalar('loss/SR', float(loss_img_each.data) * 100,
                                                         global_step=iters) # L_SR

                        self.results_recorder.add_scalar('loss/distill',
                                                         float(loss_recog_distill_each.data) * 100,
                                                         global_step=iters) # L_TP

                        self.results_recorder.add_scalar('loss/TSSIM', float(loss_tssim) * 100,
                                                         global_step=iters) # L_TSC

                    # loss_img: L_SR, L_TSC
                    # loss_recog_distill: L_TP
                    loss_im = loss_img + loss_recog_distill

                    optimizer_G.zero_grad()
                    loss_im.backward()

                    for model in model_list:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer_G.step()
                    if iters % 5 == 0:

                        self.results_recorder.add_scalar('loss/total', float(loss_im.data) * 100,
                                                    global_step=iters)

                    # torch.cuda.empty_cache()
                    if iters % cfg.displayInterval == 0:
                        print('[{}]\t'
                              'Epoch: [{}][{}/{}]\t'
                              'vis_dir={:s}\t'
                              'loss_total: {:.3f} \t'
                              'loss_im: {:.3f} \t'
                              'loss_teaching: {:.3f} \t'
                              'loss_tssim: {:.3f} \t'
                              '{:.3f} \t'
                              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                      epoch, j + 1, len(train_loader),
                                      self.vis_dir,
                                      float(loss_im.data),
                                      float(loss_img.data),
                                      float(loss_recog_distill.data),
                                      float(loss_tssim.data),
                                      lr))

                # validation & test
                if iters % cfg.VAL.valInterval == 0 or self.args.test:
                    print('======================================================')
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        for model in model_list:
                            model.eval()
                            for p in model.parameters():
                                p.requires_grad = False

                        for stu in aster_student:
                            stu.eval()
                            for p in stu.parameters():
                                p.requires_grad = False

                        # Tuned TPG for recognition:
                        # test_bible[self.args.test_model]['model'] = aster_student[-1]

                        metrics_dict = self.eval(
                            model_list=model_list,
                            val_loader=val_loader,
                            image_crit=image_crit,
                            index=iters,
                            aster=[test_bible[self.args.test_model], aster_student, aster], # aster_student[0]test_bible[self.args.test_model]
                            aster_info=aster_info,
                            data_name=data_name
                        )

                        for key in metrics_dict:
                            # print(metrics_dict)
                            if key in ["psnr_avg", "ssim_avg", "accuracy"]:
                                self.results_recorder.add_scalar('eval/' + key + "_" + data_name, float(metrics_dict[key]),
                                                    global_step=iters)

                        for stu in aster_student:
                            for p in stu.parameters():
                                p.requires_grad = True
                            stu.train()

                        for model in model_list:
                            for p in model.parameters():
                                p.requires_grad = True
                            model.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))

                        # if self.args.go_test:
                        #     break
                    if self.args.go_test:
                        break
                    if sum(current_acc_dict.values()) > best_acc:   # easy, medium, hard acc의 총합
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving BEST model at epoch %s' % str(iters))
                        # saving best checkpoints
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer=aster_student)

                # saving checkpoints for every saveInterval
                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    # print('saving model at epoch %s' % str(iters))
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, recognizer=aster_student)
            if self.args.go_test:
                break

    def eval(self, model_list, val_loader, image_crit, index, aster, aster_info, data_name=None):

        n_correct = 0
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {
            'psnr_lr': [],
            'ssim_lr': [],
            'cnt_psnr_lr': [],
            'cnt_ssim_lr': [],
            'psnr': [],
            'ssim': [],
            'cnt_psnr': [],
            'cnt_ssim': [],
            'accuracy': 0.0,
            'psnr_avg': 0.0,
            'ssim_avg': 0.0,
            'edis_LR': [],
            'edis_SR': [],
            'edis_HR': [],
            'LPIPS_VGG_LR': [],
            'LPIPS_VGG_SR': []
        }

        counters = {i: 0 for i in range(self.args.stu_iter)}
        wrong_cnt = 0
        image_counter = 0
        rec_str = ""

        sr_infer_time = 0

        #############################################
        # Print the computational cost and param size
        # self.cal_all_models(model_list, aster[1])
        #############################################

        for i, data in (enumerate(val_loader)):
            images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt = data

            # print("label_strs:", label_strs)

            images_lr = images_lrraw.to(self.device)
            images_hr = images_hrraw.to(self.device)

            val_batch_size = images_lr.shape[0]
            # images_hr = images_hr.to(self.device)

            ret_dict = self.model_inference(images_lr, images_hr, model_list, aster, i)

            # time_after = time.time()
            # print(ret_dict["duration"])
            sr_infer_time += ret_dict["duration"]
            images_sr = ret_dict["images_sr"]

            # print("images_lr:", images_lr.device, images_hr.device)

            aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
            aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])

            if self.args.test_model == "MORAN":
                # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                # LR
                aster_output_lr = aster[0]["model"](
                    aster_dict_lr[0],
                    aster_dict_lr[1],
                    aster_dict_lr[2],
                    aster_dict_lr[3],
                    test=True,
                    debug=True
                )
                # HR
                aster_output_hr = aster[0]["model"](
                    aster_dict_hr[0],
                    aster_dict_hr[1],
                    aster_dict_hr[2],
                    aster_dict_hr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_lr = aster[0]["model"](aster_dict_lr)
                aster_output_hr = aster[0]["model"](aster_dict_hr)

            if type(images_sr) == list:
                predict_result_sr = []
                for i in range(self.args.stu_iter):
                    image = images_sr[i]
                    aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])
                    if self.args.test_model == "MORAN":
                        # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                        aster_output_sr = aster[0]["model"](
                            aster_dict_sr[0],
                            aster_dict_sr[1],
                            aster_dict_sr[2],
                            aster_dict_sr[3],
                            test=True,
                            debug=True
                        )
                    else:
                        aster_output_sr = aster[0]["model"](aster_dict_sr)
                    # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                    if self.args.test_model == "CRNN":
                        predict_result_sr_ = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                    elif self.args.test_model == "ASTER":
                        predict_result_sr_, _ = aster[0]["string_process"](
                            aster_output_sr['output']['pred_rec'],
                            aster_dict_sr['rec_targets'],
                            dataset=aster_info
                        )
                    elif self.args.test_model == "MORAN":
                        preds, preds_reverse = aster_output_sr[0]
                        _, preds = preds.max(1)
                        sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                        predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]

                    predict_result_sr.append(predict_result_sr_)

                img_lr = torch.nn.functional.interpolate(images_lr, images_hr.shape[-2:], mode="bicubic")
                img_sr = torch.nn.functional.interpolate(images_sr[-1], images_hr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(img_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(img_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(img_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            else:
                images_sr = images_sr[0] if type(images_sr) == tuple else images_sr
                aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr)
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                elif self.args.test_model == "ASTER":
                    predict_result_sr, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]

                img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(images_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            if self.args.test_model == "CRNN":
                predict_result_lr = aster[0]["string_process"](aster_output_lr, self.args.CHNSR)
                predict_result_hr = aster[0]["string_process"](aster_output_hr, self.args.CHNSR)
            elif self.args.test_model == "ASTER":
                predict_result_lr, _ = aster[0]["string_process"](
                    aster_output_lr['output']['pred_rec'],
                    aster_dict_lr['rec_targets'],
                    dataset=aster_info
                )
                predict_result_hr, _ = aster[0]["string_process"](
                    aster_output_hr['output']['pred_rec'],
                    aster_dict_hr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                ### LR ###
                preds, preds_reverse = aster_output_lr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                ### HR ###
                preds, preds_reverse = aster_output_hr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

            filter_mode = 'chinese' if self.args.CHNSR else 'lower'     # case-sensitive 해야 하는거 아닌가?

            for batch_i in range(images_lr.shape[0]):

                label = label_strs[batch_i]

                image_counter += 1
                rec_str += str(image_counter) + ".jpg," + label + "\n"

                for k in range(self.args.stu_iter):
                    if str_filt(predict_result_sr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        counters[k] += 1

                if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_lr += 1
                else:
                    iswrong = True

                if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_hr += 1
                else:
                    iswrong = True

            sum_images += val_batch_size
            torch.cuda.empty_cache()

        psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['psnr']) + 1e-10)

        psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
        ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)

        lpips_vgg_lr = sum(metric_dict["LPIPS_VGG_LR"]) / (len(metric_dict['LPIPS_VGG_LR']) + 1e-10)
        lpips_vgg_sr = sum(metric_dict["LPIPS_VGG_SR"]) / (len(metric_dict['LPIPS_VGG_SR']) + 1e-10)

        print('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              'LPIPS {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), float(lpips_vgg_sr)))

        print('[{}]\t'
              'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
              'LPIPS_LR {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg_lr), float(ssim_avg_lr), float(lpips_vgg_lr)))

        # print('save display images')
        # self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)

        acc = {i: 0 for i in range(self.args.stu_iter)}
        for i in range(self.args.stu_iter):
            acc[i] = round(counters[i] / sum_images, 4)

        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)

        for i in range(self.args.stu_iter):
            print('sr_accuray_iter' + str(i) + ': %.2f%%' % (acc[i] * 100))
        accuracy = acc[self.args.stu_iter-1]

        print('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        print('hr_accuray: %.2f%%' % (accuracy_hr * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        inference_time = sum_images / sr_infer_time
        print("AVG inference:", inference_time)
        print("sum_images:", sum_images)

        return metric_dict

    def test(self):
        # model_dict = self.generator_init()
        TP_Generator_dict = {"CRNN": self.CRNN_init}
        tpg_opt = self.opt_TPG
        recognizer_path = os.path.join(self.resume, 'recognizer_best.pth')
        tpg = TP_Generator_dict[self.args.tpg](recognizer_path=recognizer_path, opt=tpg_opt)[0]
        voc_type = self.config.TRAIN.voc_type

        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data(self.test_data_dir) # self.test_data_dir: /data/gjh8760/Dataset/SITSR/TextZoom/test/easy
        data_name = self.args.test_data_dir.split('/')[-1]
        # data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
            for p in moran.parameters():
                p.requires_grad = False
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
            for p in aster.parameters():
                p.requires_grad = False
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
            for p in crnn.parameters():
                p.requires_grad = False
        # print(sum(p.numel() for p in moran.parameters()))
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        for p in tpg.parameters():
            p.requires_grad = False
        tpg.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        sr_time = 0
        for i, data in (enumerate(test_loader)):
            # images_hr, images_lr, label_strs = data
            images_hr, images_lr, _, _, label_strs, _ = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)

            label_vecs_logits = tpg(self.parse_crnn_data(images_lr[:, :3, :, :]))
            label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
            label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

            sr_begin = time.time()
            # images_sr = model(images_lr)    # 왜 TP는 사용 안하냐?
            images_sr = model(images_lr, label_vecs_final.detach())
            # images_sr: tuple.
            # images_sr[0]: sr images, images_sr[1]: attn maps
            images_sr = images_sr[0]

            # images_sr = images_lr
            sr_end = time.time()
            sr_time += sr_end - sr_begin
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                # aster_output_sr = aster(aster_dict_sr["images"])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                # pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
                pred_str_sr, _ = get_string_aster(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                # pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                pred_str_lr, _ = get_string_aster(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input["images"])
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                    n_correct += 1
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))
            # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)

            # save LR / SR / HR images, txt label
            result_img_path = os.path.join('./result/images', self.vis_dir, data_name)
            result_txt_path = os.path.join('./result/texts', self.vis_dir, data_name)
            if not os.path.exists(result_img_path):
                os.makedirs(result_img_path)
            if not os.path.exists(result_txt_path):
                os.makedirs(result_txt_path)
            for n in range(self.batch_size):
                try:
                    sr = images_sr[n][:3, :, :]  # range [0, 1]
                    lr = images_lr[n][:3, :, :]  # range [0, 1]
                    hr = images_hr[n][:3, :, :]  # range [0, 1]
                    sr = sr.permute(1, 2, 0).cpu().numpy()
                    sr = np.clip(sr * 255., 0, 255).astype(np.uint8)
                    lr = lr.permute(1, 2, 0).cpu().numpy()
                    lr = np.clip(lr * 255., 0, 255).astype(np.uint8)
                    hr = hr.permute(1, 2, 0).cpu().numpy()
                    hr = np.clip(hr * 255, 0, 255).astype(np.uint8)
                    pred_str_sr_filt = str_filt(pred_str_sr[n], voc_type)
                    label_str_filt = str_filt(label_strs[n], voc_type)
                    sr_path = os.path.join(result_img_path,
                                           f'{str(i * self.batch_size + n).zfill(4)}-sr-{pred_str_sr_filt}-{label_str_filt}.png')
                    lr_path = os.path.join(result_img_path, f'{str(i * self.batch_size + n).zfill(4)}-lr.png')
                    hr_path = os.path.join(result_img_path, f'{str(i * self.batch_size + n).zfill(4)}-hr.png')
                    imageio.imwrite(sr_path, sr)
                    imageio.imwrite(lr_path, lr)
                    imageio.imwrite(hr_path, hr)
                    with open(os.path.join(result_txt_path, f'{str(i * self.batch_size + n).zfill(4)}-sr.txt'), 'w') as f:
                        f.write(pred_str_sr[n] + '\n' + label_strs[n])
                except IndexError:
                    break

        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_begin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_begin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                     debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            print(pred_str_lr, '===>', pred_str_sr)
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)


if __name__ == '__main__':
    embed()
