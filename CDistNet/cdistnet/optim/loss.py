import torch
import torch.nn.functional as F
from torch import distributed

def cal_performance(pred, tgt, smoothing=True):
    """Returns loss and the number of correct words in training

    Args:
        pred (torch.float32) [N * tgt_len, vocab_size]: Character predicition probabilities
        tgt (torch.int64) [N, max_len] : Ground truth character indices, including pad symbols.
        smoothing (bool) : Use label smoothing.
    Returns:
        loss (torch.float32): Training loss of batch.
        n_correct (int): Total number of correctly predicted characters in batch, excluding pad symbols.
    """
    n_word = N = tgt.shape[0]
    max_len = tgt.shape[1]

    loss = cal_loss(pred, tgt, smoothing)
    pred = pred.max(1)[1]
    tgt = tgt.contiguous().view(-1)
    non_pad_mask = tgt.ne(0)
    n_char = non_pad_mask.sum().item()
    pred_eq_tgt = pred.eq(tgt)
    n_correct_char = pred_eq_tgt.masked_select(non_pad_mask).sum().item()
    
    non_pad_mask = non_pad_mask.view(N, -1)
    pred_eq_tgt = pred_eq_tgt.view(N, -1)
    n_correct_word = (pred_eq_tgt + ~non_pad_mask).sum(1).eq(max_len).sum().item()
    
    return loss, n_correct_char, n_char, n_correct_word, n_word

def cal_loss(pred, tgt, smoothing=True):
    tgt = tgt.contiguous().view(-1)
    eps = 0.1 if smoothing else 0.0
    loss = F.cross_entropy(pred, tgt, ignore_index=0, reduction='mean', label_smoothing=eps)
    return loss