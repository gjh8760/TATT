import sys
import os
import copy
from IPython import embed
import time
from datetime import datetime
import math
import shutil

import numpy as np
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
from utils.metrics import get_string_aster, get_string_crnn, get_string_cdistnet_eng, get_string_cdistnet_kor, Accuracy
from utils.util import str_filt

from model import gumbel_softmax
from loss.semantic_loss import SemanticLoss

from tensorboardX import SummaryWriter

import editdistance


sys.path.append('../')
sys.path.append('./')

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

    def torch_rotate_img(self, torch_image_batches, arc_batches, rand_offs=None, off_range=0.2):

        # ratios: H / W
        device = torch_image_batches.device

        N, C, H, W = torch_image_batches.shape
        ratios = H / float(W)

        # rand_offs = random.random() * (1 - ratios)
        if rand_offs is not None:
            ratios_mul = ratios + (rand_offs.unsqueeze(1) * off_range * 2) - off_range  # [-0.05, 0.45]
        else:
            ratios_mul = ratios

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

    def model_inference(self, images_lr, model_list, stu_model):
        ret_dict = {}   # keys: duration, images_sr
        ret_dict["duration"] = 0
        if self.args.voc_type == 'kor':
            aster_dict_lr = self.parse_cdistnet_data(images_lr[:, :3, :, :])
            before = time.time()
            label_vecs_logits = stu_model[0](aster_dict_lr, beam_size=2)    # beam size 2
            after = time.time()
        else:
            aster_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
            before = time.time()
            label_vecs_logits = stu_model[0](aster_dict_lr)
            after = time.time()

        ret_dict["duration"] += (after - before)    # TP 뽑는 데 걸리는 시간

        label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
        label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

        before = time.time()
        images_sr = model_list[0](images_lr, label_vecs_final.detach())
        after = time.time()

        # print("fps:", (after - before))
        ret_dict["duration"] += (after - before)    # SR 이미지 뽑는 데 걸리는 시간
        ret_dict["images_sr"] = images_sr
        return ret_dict

    def train(self):
        # TP_Generator_dict = {"CRNN": self.CRNN_init, "OPT": self.TPG_init}
        TP_Generator_dict = {'crnn': self.CRNN_init, 'cdistnet': self.CDistNet_init}

        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']
        curr_epoch, curr_iter = model_dict['epoch'], model_dict['iter']
        print(f"current epoch: {curr_epoch}, iter: {curr_iter}")

        model_list = [model]

        tensorboard_dir = os.path.join("tensorboard", self.vis_dir)
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        else:
            print("Directory exist, remove events...")
            os.popen("rm " + tensorboard_dir + "/*")

        self.results_recorder = SummaryWriter(tensorboard_dir)

        # English: CRNN
        # Korean: CDistNet
        stu_model_fixed, aster_info = TP_Generator_dict[self.args.tpg.lower()](recognizer_path=None)

        test_bible = {}

        if self.args.test_model.lower() == "crnn":
            crnn, aster_info = self.CRNN_init()
            for p in crnn.parameters():
                p.requires_grad = False
            crnn.eval()
            test_bible["crnn"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_crnn_data,
                        'string_process': get_string_crnn
                    }

        elif self.args.test_model.lower() == "aster":
            aster_real, aster_real_info = self.Aster_init()
            aster_info = aster_real_info
            for p in aster_real.parameters():
                p.requires_grad = False
            aster_real.eval()
            test_bible["aster"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model.lower() == "moran":
            moran = self.MORAN_init()
            for p in moran.parameters():
                p.requires_grad = False
            moran.eval()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["moran"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }

        elif self.args.test_model.lower() == 'cdistnet_kor':
            cdistnet, cdistnet_info = self.CDistNet_init()
            for p in cdistnet.parameters():
                p.requires_grad = False
            cdistnet.eval()
            test_bible['cdistnet'] = {
                'model': cdistnet,
                'data_in_fn': self.parse_cdistnet_data,
                'string_process': get_string_cdistnet
            }

        # print("self.args.arch:", self.args.arch)
        aster_student = []
        stu_iter = self.args.stu_iter

        for i in range(stu_iter):
            recognizer_path = os.path.join(self.resume, "recognizer_best.pth")
            # print("recognizer_path:", recognizer_path)
            if os.path.isfile(recognizer_path):
                # aster_student:
                aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg.lower()](recognizer_path=recognizer_path)
            else:
                aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg.lower()](recognizer_path=None)

            if type(aster_student_) == list:
                aster_student_ = aster_student_[i]

            aster_student_.train()
            aster_student.append(aster_student_)

        stu_model_fixed.eval()

        # Recognizer needs to be fixed:
        # aster
        optimizer_G = self.optimizer_init(model_list)
        # for p in aster.parameters():
        #     p.requires_grad = False

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

        for epoch in range(curr_epoch, curr_epoch + cfg.epochs):

            for j, data in (enumerate(train_loader)):
                iters = len(train_loader) * epoch + j + 1

                for model in model_list:
                    for p in model.parameters():
                        p.requires_grad = True

                # images_hrraw, images_pseudoLR, images_lrraw, images_HRy, images_lry, label_strs, label_vecs, weighted_mask, weighted_tics = data
                # images_hr, _, images_lr, _, _, _, _, _, _ = data
                if self.args.voc_type == 'kor':
                    images_hr, images_lr, _, tgt = data
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)
                    tgt = tgt.to(self.device)

                    aster_dict_hr = self.parse_cdistnet_data(images_hr[:, :3, :, :])
                    label_vecs_logits_hr = stu_model_fixed(aster_dict_hr, tgt)

                else:
                    images_hr, images_lr, _ = data
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)

                    aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :])
                    label_vecs_logits_hr = stu_model_fixed(aster_dict_hr)

                label_vecs_hr = torch.nn.functional.softmax(label_vecs_logits_hr, -1).detach()

                images_lr = images_lr.detach()

                # stu_model: extract prob. vec. from interpolated LR image, trainable
                stu_model = aster_student[0]

                if self.args.voc_type == 'kor':
                    aster_dict_lr = self.parse_cdistnet_data(images_lr[:, :3, :, :])
                    label_vecs_logits = stu_model(aster_dict_lr, tgt)
                else:
                    aster_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                    label_vecs_logits = stu_model(aster_dict_lr)

                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)

                # label_vecs_final: TP from interpolated LR image
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                # image for cascading
                pick = 0
                images_sr = model_list[pick](images_lr, label_vecs_final.detach())

                loss_img = 0.
                loss_recog_distill = 0.
                loss_tssim = torch.tensor(0.)

                # [N, C, H, W] -> [N, T, C]
                loss_recog_distill_each = sem_loss(label_vecs, label_vecs_hr) * 100 #100
                loss_recog_distill += loss_recog_distill_each  # * (1 + 0.5 * i)

                im_quality_loss = image_crit(images_sr, images_hr)

                if self.args.training_stablize:
                    im_quality_loss = self.loss_stablizing(im_quality_loss)

                loss_img_each = im_quality_loss.mean() * 100

                # loss_img += loss_img_each * (1 + i * 0.5)
                loss_img += loss_img_each

                ## loss_tssim: 1 - tssim(D(F(Y)), F(D(Y)), D(X))
                # D(F(Y)): deformed_sr
                # F(D(Y)): deformed_first_sr
                # D(X): deformed_hr
                batch_size = images_lr.shape[0]
                angle_batch = np.random.rand(batch_size) * self.args.tssim_rotation_degree * 2 - self.args.tssim_rotation_degree
                arc = torch.tensor(angle_batch / 180. * math.pi).float()
                deformed_sr = self.torch_rotate_img(images_sr, arc)
                deformed_lr = self.torch_rotate_img(images_lr, arc)
                deformed_first_sr = model_list[pick](deformed_lr, label_vecs_final.detach())
                deformed_hr = self.torch_rotate_img(images_hr, arc)
                loss_tssim = (1 - tri_ssim(deformed_sr, deformed_first_sr, deformed_hr).mean()) * 10.
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

                    self.results_recorder.add_scalar('loss/SR', float(loss_img_each.data),
                                                     global_step=iters) # L_SR

                    self.results_recorder.add_scalar('loss/distill',
                                                     float(loss_recog_distill_each.data),
                                                     global_step=iters) # L_TP

                    self.results_recorder.add_scalar('loss/TSSIM', float(loss_tssim),
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
                    self.results_recorder.add_scalar('loss/total', float(loss_im.data), global_step=iters)

                # torch.cuda.empty_cache()
                if (j+1) % cfg.displayInterval == 0:
                    print('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          'loss_total: {:.3f} \t'
                          'loss_im: {:.3f} \t'
                          'loss_teaching: {:.3f} \t'
                          'loss_tssim: {:.3f} \t'
                          '{:.4f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data),
                                  float(loss_img.data),
                                  float(loss_recog_distill.data),
                                  float(loss_tssim.data),
                                  lr))

                # validation & test
                # if iters % cfg.VAL.valInterval == 0:
                if j+1 == len(train_loader):
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
                            stu_model=aster_student,
                            test_model=test_bible[self.args.test_model.lower()],
                            aster=[test_bible[self.args.test_model.lower()], aster_student, stu_model_fixed], # aster_student[0]test_bible[self.args.test_model]
                            aster_info=aster_info
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
                # if iters % cfg.saveInterval == 0:
                if j+1 == len(train_loader):
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    # print('saving model at epoch %s' % str(iters))
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, recognizer=aster_student)

    def eval(self, model_list, val_loader, stu_model, test_model, aster, aster_info):

        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {
            'psnr_lr': [],
            'ssim_lr': [],
            'psnr': [],
            'ssim': [],
            'accuracy': 0.0,
            'psnr_avg': 0.0,
            'ssim_avg': 0.0
        }

        counters = {i: 0 for i in range(self.args.stu_iter)}
        image_counter = 0
        sr_infer_time = 0

        with torch.no_grad():

            test_parser = test_model["data_in_fn"]
            test_rec_model = test_model["model"]
            test_to_str = test_model["string_process"]

            for i, data in (enumerate(val_loader)):
                # images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt = data
                if self.args.voc_type == 'kor':
                    images_hr, images_lr, label_strs, tgt = data
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)
                    tgt = tgt.to(self.device)
                else:
                    images_hr, images_lr, label_strs = data
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)

                val_batch_size = images_lr.shape[0]

                ret_dict = self.model_inference(images_lr, model_list, stu_model)

                sr_infer_time += ret_dict["duration"]
                images_sr = ret_dict["images_sr"]

                aster_dict_lr = test_parser(images_lr[:, :3, :, :])
                aster_dict_hr = test_parser(images_hr[:, :3, :, :])
                aster_dict_sr = test_parser(images_sr[:, :3, :, :])

                if self.args.test_model.lower() == "moran":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    # LR
                    aster_output_lr = test_rec_model(
                        aster_dict_lr[0],
                        aster_dict_lr[1],
                        aster_dict_lr[2],
                        aster_dict_lr[3],
                        test=True,
                        debug=True
                    )
                    # HR
                    aster_output_hr = test_rec_model(
                        aster_dict_hr[0],
                        aster_dict_hr[1],
                        aster_dict_hr[2],
                        aster_dict_hr[3],
                        test=True,
                        debug=True
                    )
                    # SR
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                elif self.args.test_model.lower() == 'cdistnet_kor':
                    aster_output_lr = test_rec_model(aster_dict_lr, beam_size=2)
                    aster_output_hr = test_rec_model(aster_dict_hr, beam_size=2)
                    aster_output_sr = test_rec_model(aster_dict_sr, beam_size=2)
                else:
                    aster_output_lr = test_rec_model(aster_dict_lr)
                    aster_output_hr = test_rec_model(aster_dict_hr)
                    aster_output_sr = test_rec_model(aster_dict_sr)

                # predicted string result
                if self.args.test_model.lower() == "crnn":
                    predict_result_sr = test_to_str(aster_output_sr, False)
                    predict_result_lr = test_to_str(aster_output_lr, False)
                    predict_result_hr = test_to_str(aster_output_hr, False)
                elif self.args.test_model.lower() == "aster":
                    predict_result_sr, _ = test_to_str(
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                    predict_result_lr, _ = test_to_str(
                        aster_output_lr['output']['pred_rec'],
                        aster_dict_lr['rec_targets'],
                        dataset=aster_info
                    )
                    predict_result_hr, _ = test_to_str(
                        aster_output_hr['output']['pred_rec'],
                        aster_dict_hr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model.lower() == "moran":
                    ### SR ###
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]
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
                elif self.args.test_model.lower() == 'cdistnet_kor':
                    predict_result_sr = test_to_str(aster_output_sr, aster_info)
                    predict_result_lr = test_to_str(aster_output_lr, aster_info)
                    predict_result_hr = test_to_str(aster_output_hr, aster_info)

                img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))
                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                filter_mode = 'lower' if not self.args.voc_type == 'kor' else 'korean'

                for batch_i in range(images_lr.shape[0]):

                    label = label_strs[batch_i]

                    image_counter += 1

                    for k in range(self.args.stu_iter):
                        if str_filt(predict_result_sr[batch_i], filter_mode) == str_filt(label, filter_mode):
                            counters[k] += 1

                    if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct_lr += 1

                    if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct_hr += 1

                sum_images += val_batch_size
                torch.cuda.empty_cache()

            psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
            ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['psnr']) + 1e-10)

            psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
            ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)

            print('[{}]\t'
                  'loss_rec {:.3f}| loss_im {:.3f}\t'
                  'PSNR {:.2f} | SSIM {:.4f}\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          0, 0,
                          float(psnr_avg), float(ssim_avg)))

            print('[{}]\t'
                  'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          float(psnr_avg_lr), float(ssim_avg_lr)))

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
        # TP_Generator_dict = {"CRNN": self.CRNN_init}
        TP_Generator_dict = {'crnn': self.CRNN_init, 'cdistnet_kor': self.CDistNet_kor_init}
        recognizer_path = os.path.join(self.resume, 'recognizer_best.pth')

        #################   TEMP   #################
        # if self.args.kor:
        #     recognizer_path = 'models_str/cdistnet/epoch10_best_acc.pth'
        ############################################

        tpg = TP_Generator_dict[self.args.tpg.lower()](recognizer_path=recognizer_path)[0]
        voc_type = self.config.TRAIN.voc_type

        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data(self.test_data_dir) # self.test_data_dir: /data/gjh8760/Dataset/SITSR/TextZoom/test/easy
        data_name = self.args.test_data_dir.split('/')[-1]
        # data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)

        # if self.args.rec == 'moran':
        #     moran = self.MORAN_init()
        #     moran.eval()
        #     for p in moran.parameters():
        #         p.requires_grad = False
        # elif self.args.rec == 'aster':
        #     aster, aster_info = self.Aster_init()
        #     aster.eval()
        #     for p in aster.parameters():
        #         p.requires_grad = False
        # elif self.args.rec == 'crnn':
        #     crnn = self.CRNN_init()
        #     crnn.eval()
        #     for p in crnn.parameters():
        #         p.requires_grad = False

        recognition_model, info = self.init_test_recognition_model()    # info is only for aster model

        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        for p in tpg.parameters():
            p.requires_grad = False
        tpg.eval()
        n_correct = 0
        n_correct_lr = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        metric_dict_lr = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        current_acc_dict_lr = {data_name: 0}
        sr_time = 0
        for i, data in (enumerate(test_loader)):
            if self.args.voc_type == 'kor':
                images_hr, images_lr, label_strs, _ = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                input_for_rec_model = self.parse_cdistnet_data(images_lr[:, :3, :, :])
                label_vecs_logits = tpg(input_for_rec_model, beam_size=2)   # T, B, K   # TODO: Test에서 TP 뽑을 때 beam_size = 1 or 2?

            else:
                images_hr, images_lr, label_strs = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                input_for_rec_model = self.parse_crnn_data(images_lr[:, :3, :, :])
                label_vecs_logits = tpg(input_for_rec_model)

            val_batch_size = images_lr.shape[0]

            label_vecs = F.softmax(label_vecs_logits, -1)   # T, B, K
            label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)     # B, K, 1, T

            sr_begin = time.time()
            images_sr = model(images_lr, label_vecs_final.detach())

            sr_end = time.time()
            sr_time += sr_end - sr_begin

            # HR metric
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            # bicubic LR metric
            images_upsampled_lr = F.interpolate(images_lr, (32, 128), mode='bicubic').clamp(0., 1.)
            metric_dict_lr['psnr'].append(self.cal_psnr(images_upsampled_lr, images_hr))
            metric_dict_lr['ssim'].append(self.cal_ssim(images_upsampled_lr, images_hr))

            pred_str_sr = self.get_recognition_result(recognition_model=recognition_model,
                                                      input_images=images_sr[:, :3, :, :],
                                                      dataset_info=info)
            pred_str_lr = self.get_recognition_result(recognition_model=recognition_model,
                                                      input_images=images_lr[:, :3, :, :],
                                                      dataset_info=info)

            is_cdistnet_eng = True if self.args.test_model.lower() == 'cdistnet_eng' else False
            # HR # of correct samples
            for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, voc_type, is_cdistnet_eng) == str_filt(target, voc_type, is_cdistnet_eng):
                    n_correct += 1

            # LR # of correct samples
            for pred, target in zip(pred_str_lr, label_strs):
                if str_filt(pred, voc_type, is_cdistnet_eng) == str_filt(target, voc_type, is_cdistnet_eng):
                    n_correct_lr += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))

            # save LR / SR / HR images, txt label
            result_img_path = os.path.join('./result/images', self.vis_dir, data_name)
            result_txt_path = os.path.join('./result/texts', self.vis_dir, data_name)
            ## for mispredicted samples
            result_wrong_img_path = os.path.join('./result/images_wrong', self.vis_dir, data_name)
            result_wrong_txt_path = os.path.join('./result/texts_wrong', self.vis_dir, data_name)
            ## for correctly predicted, but not predicted in LR
            result_better_img_path = os.path.join('./result/images_better', self.vis_dir, data_name)
            result_better_txt_path = os.path.join('./result/texts_better', self.vis_dir, data_name)
            ## for correctly predicted with LR, but not predicted with SR
            result_worse_img_path = os.path.join('./result/images_worse', self.vis_dir, data_name)
            result_worse_txt_path = os.path.join('./result/texts_wrose', self.vis_dir, data_name)

            path_list = [result_img_path, result_txt_path, result_wrong_img_path, result_wrong_txt_path,
                         result_better_img_path, result_better_txt_path, result_worse_img_path, result_worse_txt_path]

            for path in path_list:
                if i == 0:
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    os.makedirs(path)

            for n in range(self.batch_size):
                try:
                    sr = images_sr[n][:3, :, :]  # range [0, 1]
                    lr = images_lr[n][:3, :, :]  # range [0, 1]
                    hr = images_hr[n][:3, :, :]  # range [0, 1]
                    bi = F.interpolate(lr.unsqueeze(0), (32, 128), mode='bicubic').clamp(0., 1.).squeeze(0)
                    sr = sr.permute(1, 2, 0).cpu().numpy()
                    sr = np.clip(sr * 255., 0, 255).astype(np.uint8)
                    lr = lr.permute(1, 2, 0).cpu().numpy()
                    lr = np.clip(lr * 255., 0, 255).astype(np.uint8)
                    hr = hr.permute(1, 2, 0).cpu().numpy()
                    hr = np.clip(hr * 255, 0, 255).astype(np.uint8)
                    bi = bi.permute(1, 2, 0).cpu().numpy()
                    bi = np.clip(bi * 255, 0, 255).astype(np.uint8)
                    pred_str_lr_filt = str_filt(pred_str_lr[n], voc_type, is_cdistnet_eng)
                    pred_str_sr_filt = str_filt(pred_str_sr[n], voc_type, is_cdistnet_eng)
                    label_str_filt = str_filt(label_strs[n], voc_type, is_cdistnet_eng)
                    sr_path = os.path.join(result_img_path,
                                           f'{str(i * self.batch_size + n).zfill(4)}-sr-{pred_str_sr_filt}-{label_str_filt}.png')
                    lr_path = os.path.join(result_img_path, f'{str(i * self.batch_size + n).zfill(4)}-lr.png')
                    hr_path = os.path.join(result_img_path, f'{str(i * self.batch_size + n).zfill(4)}-hr.png')
                    bi_path = os.path.join(result_img_path, f'{str(i * self.batch_size + n).zfill(4)}-bi.png')
                    imageio.imwrite(sr_path, sr)
                    imageio.imwrite(lr_path, lr)
                    imageio.imwrite(hr_path, hr)
                    imageio.imwrite(bi_path, bi)
                    with open(os.path.join(result_txt_path, f'{str(i * self.batch_size + n).zfill(4)}-sr.txt'), 'w') as f:
                        f.write(pred_str_sr[n] + '\n' + label_strs[n])

                    # Save mispredicted results
                    if pred_str_sr_filt != label_str_filt:
                        sr_wrong_path = os.path.join(result_wrong_img_path,
                                                     f'{str(i * self.batch_size + n).zfill(4)}-sr-{pred_str_sr_filt}-{label_str_filt}.png')
                        lr_wrong_path = os.path.join(result_wrong_img_path, f'{str(i * self.batch_size + n).zfill(4)}-lr.png')
                        hr_wrong_path = os.path.join(result_wrong_img_path, f'{str(i * self.batch_size + n).zfill(4)}-hr.png')
                        bi_wrong_path = os.path.join(result_wrong_img_path, f'{str(i * self.batch_size + n).zfill(4)}-bi.png')
                        imageio.imwrite(sr_wrong_path, sr)
                        imageio.imwrite(lr_wrong_path, lr)
                        imageio.imwrite(hr_wrong_path, hr)
                        imageio.imwrite(bi_wrong_path, bi)
                        with open(os.path.join(result_wrong_txt_path, f'{str(i * self.batch_size + n).zfill(4)}-sr.txt'),
                                  'w') as f:
                            f.write(pred_str_sr[n] + '\n' + label_strs[n])

                    # Save LR mispred but SR correctly pred
                    if pred_str_lr_filt != label_str_filt and pred_str_sr_filt == label_str_filt:
                        sr_better_path = os.path.join(result_better_img_path,
                                                      f'{str(i * self.batch_size + n).zfill(4)}-sr-{pred_str_sr_filt}-{label_str_filt}.png')
                        lr_better_path = os.path.join(result_better_img_path,
                                                      f'{str(i * self.batch_size + n).zfill(4)}-lr-{pred_str_lr_filt}-{label_str_filt}.png')
                        hr_better_path = os.path.join(result_better_img_path, f'{str(i * self.batch_size + n).zfill(4)}-hr.png')
                        bi_better_path = os.path.join(result_better_img_path, f'{str(i * self.batch_size + n).zfill(4)}-bi.png')
                        imageio.imwrite(sr_better_path, sr)
                        imageio.imwrite(lr_better_path, lr)
                        imageio.imwrite(hr_better_path, hr)
                        imageio.imwrite(bi_better_path, bi)
                        with open(os.path.join(result_better_txt_path, f'{str(i * self.batch_size + n).zfill(4)}-sr.txt'),
                                  'w') as f:
                            f.write(pred_str_sr[n] + '\n' + label_strs[n])

                    # Save results which got worse accuracy after SR
                    if pred_str_lr_filt == label_str_filt and pred_str_sr_filt != label_str_filt:
                        sr_worse_path = os.path.join(result_worse_img_path,
                                                     f'{str(i * self.batch_size + n).zfill(4)}-sr-{pred_str_sr_filt}-{label_str_filt}.png')
                        lr_worse_path = os.path.join(result_worse_img_path,
                                                     f'{str(i * self.batch_size + n).zfill(4)}-lr-{pred_str_lr_filt}-{label_str_filt}.png')
                        hr_worse_path = os.path.join(result_worse_img_path, f'{str(i * self.batch_size + n).zfill(4)}-hr.png')
                        bi_worse_path = os.path.join(result_worse_img_path, f'{str(i * self.batch_size + n).zfill(4)}-bi.png')
                        imageio.imwrite(sr_worse_path, sr)
                        imageio.imwrite(lr_worse_path, lr)
                        imageio.imwrite(hr_worse_path, hr)
                        imageio.imwrite(bi_worse_path, bi)
                        with open(os.path.join(result_worse_txt_path, f'{str(i * self.batch_size + n).zfill(4)}-sr.txt'),
                                  'w') as f:
                            f.write(pred_str_sr[n] + '\n' + label_strs[n])

                except IndexError:
                    break

        # SR result
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg}
        print('SR: ', result)

        # LR result
        psnr_avg_lr = sum(metric_dict_lr['psnr']) / len(metric_dict_lr['psnr'])
        ssim_avg_lr = sum(metric_dict_lr['ssim']) / len(metric_dict_lr['ssim'])
        acc_lr = round(n_correct_lr / sum_images, 4)
        psnr_avg_lr = round(psnr_avg_lr.item(), 6)
        ssim_avg_lr = round(ssim_avg_lr.item(), 6)
        current_acc_dict_lr[data_name] = float(acc_lr)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result_lr = {'accuracy': current_acc_dict_lr, 'psnr_avg': psnr_avg_lr, 'ssim_avg': ssim_avg_lr}
        print('Bicubic LR: ', result_lr)

    def init_test_recognition_model(self):
        info = None
        if self.args.test_model.lower() == 'moran':
            recognition_model = self.MORAN_init()
        elif self.args.test_model.lower() == 'aster':
            recognition_model, info = self.Aster_init()
        elif self.args.test_model.lower() == 'crnn':
            recognition_model = self.CRNN_init()
        elif self.args.test_model.lower() == 'cdistnet_eng':
            recognition_model, info = self.CDistNet_eng_init()
        elif self.args.test_model.lower() == 'cdistnet_kor':
            recognition_model, info = self.CDistNet_kor_init()
        recognition_model.eval()
        for p in recognition_model.parameters():
            p.requires_grad = False
        return recognition_model, info

    def get_recognition_result(self, recognition_model, input_images, dataset_info):
        # input_images: 3 channel images, without mask channel
        if self.args.test_model.lower() == 'moran':
            model_input = self.parse_moran_data(input_images)
            model_output = recognition_model(model_input[0], model_input[1], model_input[2], model_input[3], test=True,
                                             debug=True)
            preds, preds_reverse = model_output[0]
            _, preds = preds.max(1)
            sim_preds = self.converter_moran.decode(preds.data, model_input[1].data)
            pred_str = [pred.split('$')[0] for pred in sim_preds]
        elif self.args.test_model.lower() == 'aster':
            model_input = self.parse_aster_data(input_images)
            model_output = recognition_model(model_input)
            pred_rec = model_output['output']['pred_rec']
            pred_str, _ = get_string_aster(pred_rec, model_input['rec_targets'], dataset=dataset_info)
        elif self.args.test_model.lower() == 'crnn':
            model_input = self.parse_crnn_data(input_images)
            model_output = recognition_model(model_input['images'])
            _, preds = model_output.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = torch.IntTensor([model_output.size(0)] * input_images.shape[0])
            pred_str = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
        elif self.args.test_model.lower() == 'cdistnet_eng':
            model_input = self.parse_cdistnet_data(input_images)
            model_output = recognition_model(model_input, beam_size=2)
            pred_str = get_string_cdistnet_eng(output=model_output, dataset=dataset_info)
        elif self.args.test_model.lower() == 'cdistnet_kor':
            model_input = self.parse_cdistnet_data(input_images)
            model_output = recognition_model(model_input, beam_size=2)      # beam size 2
            pred_str = get_string_cdistnet_kor(output=model_output, dataset=dataset_info)
        return pred_str


if __name__ == '__main__':
    embed()
