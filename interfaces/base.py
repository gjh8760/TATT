import torch
import sys
import os
import torch.optim as optim
import string
from collections import OrderedDict

import ptflops

from model import tsrn
from model import recognizer
from model import moran
from model import crnn

from dataset import lmdbDataset, alignCollate
from dataset import lmdbDataset_kor, alignCollate_kor
from dataset import lmdbDataset_eng, alignCollate_eng
from dataset import lmdbDataset_all, alignCollate_all
from loss import gradient_loss, percptual_loss, image_loss, semantic_loss

from utils.labelmaps import get_vocabulary, labels2strs

sys.path.append('../')
from utils import util, ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset

from model.cdistnet.model import build_CDistNet

from mmcv import Config


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        if self.args.voc_type == 'kor':
            # self.config_CDistNet = 'CDistNet/ckpts/yeti_2022-07-11/CDistNet_config_kor.py'
            self.config_CDistNet = 'models_str/cdistnet_kor/CDistNet_config_kor.py'
        elif self.args.voc_type == 'eng':
            # self.config_CDistNet = 'CDistNet/ckpts/yeti_2022-04-13/CDistNet_config.py'
            self.config_CDistNet = 'models_str/cdistnet_eng/CDistNet_config.py'
        elif self.args.voc_type == 'all':
            # self.config_CDistNet = 'CDistNet/ckpts/v100_2022-04-13/CDistNet_config_v100.py'
            self.config_CDistNet = 'models_str/cdistnet_all/CDistNet_config_v100.py'
        self.scale_factor = self.config.TRAIN.down_sample_scale

        if self.args.voc_type == 'kor':
            self.align_collate = alignCollate_kor
            self.load_dataset = lmdbDataset_kor
            self.align_collate_val = alignCollate_kor
            self.load_dataset_val = lmdbDataset_kor
        elif self.args.voc_type == 'eng' and self.args.tpg == 'cdistnet_eng':
            self.align_collate = alignCollate_eng
            self.load_dataset = lmdbDataset_eng
            self.align_collate_val = alignCollate_eng
            self.load_dataset_val = lmdbDataset_eng
        elif self.args.voc_type == 'all':
            self.align_collate = alignCollate_all
            self.load_dataset = lmdbDataset_all
            self.align_collate_val = alignCollate_all
            self.load_dataset_val = lmdbDataset_all
        else:   # voc_type == eng, tpg == crnn
            self.align_collate = alignCollate
            self.load_dataset = lmdbDataset
            self.align_collate_val = alignCollate
            self.load_dataset_val = lmdbDataset

        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation,
        }
        # korean
        voc_korean_list = open('./CDistNet/cdistnet/utils/dict_2354_kor.txt', 'r').readlines()
        voc_korean_list = [x.replace('\n', '') for x in voc_korean_list]
        alpha_dict['korean'] = voc_korean_list
        # cdistnet_eng
        if self.args.tpg == 'cdistnet_eng':
            voc_eng_list = open('./CDistNet/cdistnet/utils/dict_40_num_eng.txt', 'r').readlines()
            voc_eng_list = [x.replace('\n', '') for x in voc_eng_list]
            alpha_dict['lower'] = voc_eng_list
        # cdistnet_all
        if self.args.tpg == 'cdistnet_all':
            voc_all_list = open('./CDistNet/cdistnet/utils/dict_2448_num_eng_Eng_spe_kor.txt', 'r').readlines()
            voc_all_list = [x.replace('\n', '') for x in voc_all_list]
            alpha_dict['all'] = voc_all_list
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.word2idx = {word: idx for idx, word in enumerate(self.alphabet)}
        self.idx2word = {idx: word for idx, word in enumerate(self.alphabet)}
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.ckpt_path = os.path.join('ckpt', self.vis_dir)
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.cal_psnr_weighted = ssim_psnr.weighted_calculate_psnr
        self.cal_ssim_weighted = ssim_psnr.SSIM_WEIGHTED()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len,
                                      word2idx=self.word2idx
                ))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        # print("cfg.down_sample_scale:", cfg.down_sample_scale)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, # int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=True),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        # self.args.test_data_dir
        test_dataset = self.load_dataset_val(root=dir_,  #load_dataset
                                             voc_type=cfg.voc_type,
                                             max_len=cfg.max_len,
                                             word2idx=self.word2idx
                                             )

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=0, # int(cfg.workers),
                                                  collate_fn=self.align_collate_val(imgH=cfg.height,
                                                                                    imgW=cfg.width,
                                                                                    down_sample_scale=cfg.down_sample_scale,
                                                                                    mask=self.mask,
                                                                                    train=False),
                                                  drop_last=False)
        return test_dataset, test_loader

    def generator_init(self, iter=-1, resume_in=None):
        cfg = self.config.TRAIN

        resume = self.resume
        if not resume_in is None:
            resume = resume_in

        if self.args.voc_type == 'kor':
            text_embedding = 2354
        elif self.args.voc_type == 'eng' and self.args.tpg == 'cdistnet_eng':
            text_embedding = 40
        elif self.args.voc_type == 'all':
            text_embedding = 2448
        else:   # voc_type == eng, tpg == crnn
            text_embedding = 37

        model = tsrn.TSRN_TL_TRANS(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                   STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                   hidden_units=self.args.hd_u, text_embedding=text_embedding)
        image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        channel_size = 4
        macs, params = ptflops.get_model_complexity_info(model, (channel_size, 16, 64), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- SR Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        model = model.to(self.device)
        image_crit.to(self.device)

        epoch = 0
        iters = 0

        if not resume == '':
            print('loading pre-trained model from %s ' % resume)
            # if is dir, we need to initialize the model list
            if os.path.isdir(resume):
                print("resume:", resume)
                ckpt = torch.load(os.path.join(resume, "model_best.pth"))
                model_dict = ckpt['state_dict_G']
                model.load_state_dict(model_dict, strict=False)
                epoch = ckpt['info']['epochs']
                iters = ckpt['info']['iters']

            else:
                ckpt = torch.load(resume)
                model.load_state_dict(ckpt['state_dict_G'])
                epoch = ckpt['info']['epochs']
                iters = ckpt['info']['iters']

        return {'model': model, 'crit': image_crit, 'epoch': epoch, 'iter': iters}

    def optimizer_init(self, model_list, learning_rate_list):
        cfg = self.config.TRAIN

        # Set learning rate for each model
        param_lr_list = []
        for idx, model in enumerate(model_list):
            param_lr_dict = {}
            lr = learning_rate_list[idx]
            model_params = []
            for m in model:
                model_params += list(m.parameters())
            param_lr_dict['params'] = model_params
            param_lr_dict['lr'] = lr
            param_lr_list.append(param_lr_dict)

        if cfg.optimizer == 'Adam':
            optimizer = optim.Adam(param_lr_list, lr=cfg.lr, betas=(cfg.beta1, 0.999))
        if cfg.optimizer == 'SGD':
            optimizer = optim.SGD(param_lr_list, lr=cfg.lr, momentum=0.9)

        return optimizer

    def save_checkpoint(self, netG_list, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, recognizer=None, prefix="acc", global_model=None):

        ckpt_path = self.ckpt_path# = os.path.join('ckpt', self.vis_dir)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        netG_ = netG_list[0]

        if self.config.TRAIN.ngpu > 1:
            netG = netG_.module
        else:
            netG = netG_

        save_dict = {
            'state_dict_G': netG.state_dict(),
            'info': {'arch': 'tatt', 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in netG.parameters()]),
            'converge': converge_list,
        }

        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
            if not recognizer is None:
                rec_state_dict = recognizer[0].state_dict()
                torch.save(rec_state_dict,
                           os.path.join(ckpt_path, 'recognizer_best.pth'))
            if not global_model is None:
                torch.save(global_model, os.path.join(ckpt_path, 'global_model_best.pth'))

        torch.save(save_dict, os.path.join(ckpt_path, 'model_' + str(iters) + '.pth'))
        if not recognizer is None:
            torch.save(recognizer[0].state_dict(),
                       os.path.join(ckpt_path, 'recognizer_' + str(iters) + '.pth'))
        if not global_model is None:
            torch.save(global_model.state_dict(), os.path.join(ckpt_path, 'global_model_' + str(iters) + '.pth'))


    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self, recognizer_path=None):
        model = crnn.CRNN(imgH=32, nc=1, nclass=37, nh=256)
        model = model.to(self.device)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        print("recognizer_path:", recognizer_path)

        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        if recognizer_path is None:
            model.load_state_dict(stat_dict)
        else:
            # print("stat_dict:", stat_dict)
            # print("stat_dict:", type(stat_dict) == OrderedDict)
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def CDistNet_eng_init(self, recognizer_path=None):
        """
        Initialize CDistNet_eng parameters.
        """
        cfg = Config.fromfile(self.config_CDistNet)
        model = build_CDistNet(cfg)

        cdistnet_info = CDistNetInfo(voc_type='lower')
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.cdistnet_eng_pretrained
        print("recognizer_path:", model_path)
        print('loading pretrained CDistNet model from %s' % model_path)
        stat_dict = torch.load(model_path, map_location=self.device)

        try:
            model.load_state_dict(stat_dict['model'])
        except:
            model.load_state_dict(stat_dict)
        model.to(self.device)

        return model, cdistnet_info

    def CDistNet_kor_init(self, recognizer_path=None):
        """
        Initialize CDistNet_kor parameters.
        """
        cfg = Config.fromfile(self.config_CDistNet)
        model = build_CDistNet(cfg)

        cdistnet_info = CDistNetInfo(voc_type='korean')
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.cdistnet_kor_pretrained
        print("recognizer_path:", model_path)
        print('loading pretrained CDistNet model from %s' % model_path)
        stat_dict = torch.load(model_path, map_location=self.device)

        try:
            model.load_state_dict(stat_dict['model'])
        except:
            model.load_state_dict(stat_dict)
        model.to(self.device)

        return model, cdistnet_info

    def CDistNet_all_init(self, recognizer_path=None):
        """
        Initialize CDistNet_all parameters.
        """
        cfg = Config.fromfile(self.config_CDistNet)
        model = build_CDistNet(cfg)

        cdistnet_info = CDistNetInfo(voc_type='all')
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.cdistnet_all_pretrained
        print("recognizer_path:", model_path)
        print('loading pretrained CDistNet model from %s' % model_path)
        stat_dict = torch.load(model_path, map_location=self.device)

        try:
            model.load_state_dict(stat_dict['model'])
        except:
            model.load_state_dict(stat_dict)
        model.to(self.device)

        return model, cdistnet_info

    def TPG_init(self, recognizer_path=None, opt=None):
        model = crnn.Model(opt)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else opt.saved_model
        print('loading pretrained TPG model from %s' % model_path)
        stat_dict = torch.load(model_path)

        model_keys = model.state_dict().keys()
        if type(stat_dict) == list:
            print("state_dict:", len(stat_dict))
            stat_dict = stat_dict[0]#.state_dict()

        if recognizer_path is None:
            load_keys = stat_dict.keys()
            man_load_dict = model.state_dict()
            for key in stat_dict:
                if not key.replace("module.", "") in man_load_dict:
                    print("Key not match", key, key.replace("module.", ""))
                man_load_dict[key.replace("module.", "")] = stat_dict[key]
            model.load_state_dict(man_load_dict)
        else:
            model.load_state_dict(stat_dict)

        return model, aster_info

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def parse_cdistnet_data(self, imgs_input_):
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, 128), mode='bicubic').clamp(0., 1.)
        return imgs_input * 2. - 1.

    def parse_OPT_data(self, imgs_input_, ratio_keep=False):

        in_width = 512

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        aster_info = AsterInfo('all')   # cfg.voc_type
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('load pretrained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        # aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        aster.eval()
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        aster_info = AsterInfo('all')    # cfg.voc_type
        input_dict = {}
        images_input = imgs_input.to(self.device)
        images_input = torch.nn.functional.interpolate(images_input, (32, 128), mode='bicubic')
        input_dict['images'] = images_input * 2. - 1.
        batch_size = images_input.shape[0]
        # input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1).to(self.device)    # 왜 1로 채움?
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese', 'korean']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)


class CDistNetInfo(object):
    def __init__(self, voc_type):
        self.voc_type = voc_type
        assert self.voc_type in ['lower', 'korean', 'all']

        if self.voc_type == 'korean':
            self.max_len = 26
            voc = open('CDistNet/cdistnet/utils/dict_2354_kor.txt', 'r').readlines()
        elif self.voc_type == 'lower':
            self.max_len = 100
            voc = open('CDistNet/cdistnet/utils/dict_40_num_eng.txt', 'r').readlines()
        else:   # self.voc_type == 'all'
            self.max_len = 100
            voc = open('CDistNet/cdistnet/utils/dict_2448_num_eng_Eng_spe_kor.txt', 'r').readlines()

        self.voc = [x.replace('\n', '') for x in voc]
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
