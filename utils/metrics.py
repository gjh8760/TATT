from __future__ import absolute_import

import numpy as np
import editdistance
import string
import math
from IPython import embed
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils import to_torch, to_numpy


def _normalize_text(text):
    text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
    return text.lower()


def get_string_aster(output, target, dataset=None):
    # label_seq
    assert output.dim() == 2 and target.dim() == 2

    end_label = dataset.char2id[dataset.EOS]
    unknown_label = dataset.char2id[dataset.UNKNOWN]
    num_samples, max_len_labels = output.size()
    num_classes = len(dataset.char2id.keys())
    assert num_samples == target.size(0) and max_len_labels == target.size(1)
    output = to_numpy(output)
    target = to_numpy(target)

    # list of char list
    pred_list, targ_list = [], []
    for i in range(num_samples):
        pred_list_i = []
        for j in range(max_len_labels):
            if output[i, j] != end_label:
                if output[i, j] != unknown_label:
                    try:
                        pred_list_i.append(dataset.id2char[output[i, j]])
                    except:
                        embed(header='problem')
            else:
                break
        pred_list.append(pred_list_i)

    for i in range(num_samples):
        targ_list_i = []
        for j in range(max_len_labels):
            if target[i, j] != end_label:
                if target[i, j] != unknown_label:
                    targ_list_i.append(dataset.id2char[target[i, j]])
            else:
                break
        targ_list.append(targ_list_i)

    # char list to string
    # if dataset.lowercase:
    if True:
        # pred_list = [''.join(pred).lower() for pred in pred_list]
        # targ_list = [''.join(targ).lower() for targ in targ_list]
        pred_list = [_normalize_text(pred) for pred in pred_list]
        targ_list = [_normalize_text(targ) for targ in targ_list]
    else:
        pred_list = [''.join(pred) for pred in pred_list]
        targ_list = [''.join(targ) for targ in targ_list]

    return pred_list, targ_list


def get_string_cdistnet_eng(output, dataset=None):
    # output: T, B, K
    _, pred_rec = output.topk(1, dim=2)
    pred_rec = to_numpy(pred_rec.squeeze(2))  # T, B

    # list of char list
    pred_list = []
    seq_len, batch_size, num_classes = output.shape
    for b in range(batch_size):
        pred_str_b = ''
        for t in range(seq_len):
            if pred_rec[t, b] != 3:     # 3: eos
                pred_str_b += dataset.id2char[pred_rec[t, b]]
            else:
                break
        pred_list.append(pred_str_b)
    return pred_list


def get_string_cdistnet_kor(output, dataset=None):
    # output: T, B, K
    _, pred_rec = output.topk(1, dim=2)
    pred_rec = to_numpy(pred_rec.squeeze(2))  # T, B

    # list of char list
    pred_list = []
    seq_len, batch_size, num_classes = output.shape
    for b in range(batch_size):
        pred_str_b = ''
        for t in range(seq_len):
            if pred_rec[t, b] != 3:     # 3: eos
                pred_str_b += dataset.id2char[pred_rec[t, b]]
            else:
                break
        pred_list.append(pred_str_b)
    return pred_list


def get_string_cdistnet_all(output, dataset=None):
    # output: T, B, K
    _, pred_rec = output.topk(1, dim=2)
    pred_rec = to_numpy(pred_rec.squeeze(2))  # T, B

    # list of char list
    pred_list = []
    seq_len, batch_size, num_classes = output.shape
    for b in range(batch_size):
        pred_str_b = ''
        for t in range(seq_len):
            if pred_rec[t, b] != 3:     # 3: eos
                pred_str_b += dataset.id2char[pred_rec[t, b]]
            else:
                break
        pred_list.append(pred_str_b)
    return pred_list


def get_string_crnn(outputs_, use_chinese=False, alphabet='-0123456789abcdefghijklmnopqrstuvwxyz'):
    outputs = outputs_.permute(1, 0, 2).contiguous()
    predict_result = []

    for output in outputs:
        max_index = torch.max(output, 1)[1]

        out_str = ""
        last = ""
        for i in max_index:
            if alphabet[i] != last:
                if i != 0:
                    out_str += alphabet[i]
                    last = alphabet[i]
                else:
                    last = ""

        predict_result.append(out_str)
    return predict_result


def _lexicon_search(lexicon, word):
    edit_distances = []
    for lex_word in lexicon:
        edit_distances.append(editdistance.eval(_normalize_text(lex_word), _normalize_text(word)))
    edit_distances = np.asarray(edit_distances, dtype=np.int)
    argmin = np.argmin(edit_distances)
    return lexicon[argmin]


def Accuracy(output, target, dataset=None):
    pred_list, targ_list = get_string_aster(output, target, dataset)

    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    return accuracy


def Accuracy_with_lexicon(output, target, dataset=None, file_names=None):
    pred_list, targ_list = get_string_aster(output, target, dataset)
    accuracys = []

    # with no lexicon
    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    accuracys.append(accuracy)

    # lexicon50
    if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons50[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)

    # lexicon1k
    if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons1k[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)

    # lexiconfull
    if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexiconsfull[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)

    return accuracys


def EditDistance(output, target, dataset=None):
    pred_list, targ_list = get_string_aster(output, target, dataset)

    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(pred_list, targ_list)]
    eds = sum(ed_list)
    return eds


def EditDistance_with_lexicon(output, target, dataset=None, file_names=None):
    pred_list, targ_list = get_string_aster(output, target, dataset)
    eds = []

    # with no lexicon
    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(pred_list, targ_list)]
    ed = sum(ed_list)
    eds.append(ed)

    # lexicon50
    if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
        eds.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons50[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refined_pred_list, targ_list)]
        ed = sum(ed_list)
        eds.append(ed)

    # lexicon1k
    if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
        eds.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons1k[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refined_pred_list, targ_list)]
        ed = sum(ed_list)
        eds.append(ed)

    # lexiconfull
    if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
        eds.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexiconsfull[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refined_pred_list, targ_list)]
        ed = sum(ed_list)
        eds.append(ed)

    return eds


def RecPostProcess(output, target, score, dataset=None):
    pred_list, targ_list = get_string_aster(output, target, dataset)
    max_len_labels = output.size(1)
    score_list = []

    score = to_numpy(score)
    for i, pred in enumerate(pred_list):
        len_pred = len(pred) + 1 # eos should be included
        len_pred = min(max_len_labels, len_pred) # maybe the predicted string don't include a eos.
        score_i = score[i,:len_pred]
        score_i = math.exp(sum(map(math.log, score_i)))
        score_list.append(score_i)
    return pred_list, targ_list, score_list