import torch
import torch.nn.functional as F


class Beam:
    ''' Beam search '''

    def __init__(self, beam_size, device=False):
        self.beam_size = beam_size
        self._done = False

        # The score for each translation on the beam.
        self.scores = None
        self.all_scores = []
        self.all_features = []

        # The backpointers at each time-step.
        self.parent_nodes = []

        # The outputs at each time-step.
        # self.nodes = [torch.full((beam_size,), 0, dtype=torch.long, device=device)]     # TODO: 0 말고 2로 초기화해야 하지 않나?
        self.nodes = [torch.full((beam_size,), 2, dtype=torch.long, device=device)]
        self.nodes[0][0] = 2

    @property
    def done(self):
        # Beam condition.
        # Cannot be set from outsize of class.
        return self._done

    def sort_scores(self):
        """Sort scores in descending order"""
        return torch.sort(self.scores, 0, True)

    def get_current_origin(self):
        """Get the backpointers for the current timestep"""
        return self.parent_nodes[-1]

    def update(self, char_prob, logit):
        """Update beams' status to current time step and check if finished or not

        Args:
            char_prob (torch.float32, [beam_size, vocab_size]) : Predicted character probability scores of current time step.

        Updates:
            self.scores : Update to current best_scores
            self.parent_nodes : Append
        """
        vocab_size = char_prob.size(1)

        # Sum previous scores.
        if len(self.parent_nodes) > 0:
            beam_candidates = char_prob + self.scores.unsqueeze(1).expand_as(char_prob)
        else:
            beam_candidates = char_prob[0]

        flat_beam_candidates = beam_candidates.view(-1)

        best_scores, best_scores_candidate_indices = flat_beam_candidates.topk(self.beam_size, dim=0, largest=True,
                                                                               sorted=True)
        self.scores = best_scores
        self.all_scores.append(self.scores)

        node_parent_indices = best_scores_candidate_indices // vocab_size
        self.parent_nodes.append(node_parent_indices)  # 0 ~ beam_size

        # Append logit
        self.all_features.append([logit[i] for i in node_parent_indices])

        node_indices = best_scores_candidate_indices - node_parent_indices * vocab_size  # 0 ~ vocab_size
        self.nodes.append(node_indices)

        # End condition is when top-of-beam is EOS.
        if self.nodes[-1][0].item() == 3:
            self._done = True

        return self._done

    def get_hypothesis(self):
        """ Get the decoded sequence for all beams. """
        if len(self.nodes) == 1:
            dec_seq = self.nodes[0].unsqueeze(1)
        else:
            _, node_indices = self.sort_scores()
            hyps = [self.get_beam_hypothesis(node_idx) for node_idx in node_indices]
            hyps = [[2] + h for h in hyps]  # SOS
            dec_seq = torch.LongTensor(hyps)
        return dec_seq

    def get_beam_hypothesis(self, node_idx):
        """ Get the decoded sequence for a specific beam. """
        hyp = []
        for j in reversed(range(len(self.parent_nodes))):
            timestep = j + 1
            hyp.append(self.nodes[timestep][node_idx])
            node_idx = self.parent_nodes[timestep - 1][node_idx]
        return list(map(lambda x: x.item(), hyp[::-1]))


def get_batch_idx_to_beam_idx(active_batch_indices):
    return {active_batch_idx: beam_idx for beam_idx, active_batch_idx in enumerate(active_batch_indices)}


def beam_decode_step(
        model,
        b_beams,
        timestep,
        enc_output,
        batch_idx_to_beam_idx,
        beam_size,
        vis_key_padding_mask
):
    ''' Decode and update beam status, and then return active beam idx '''
    n_active_beams = len(batch_idx_to_beam_idx)
    dec_seq = initialize_beam_dec_seq(b_beams, timestep)
    vis_key_padding_mask = None         # keep_aspect_ratio = False
    char_prob, logit = predict_char(model, dec_seq, enc_output, n_active_beams, beam_size, vis_key_padding_mask)
    active_batch_indices = update_beams(b_beams=b_beams, char_prob=char_prob, logit=logit, batch_idx_to_beam_idx=batch_idx_to_beam_idx)
    return active_batch_indices, char_prob


def initialize_beam_dec_seq(b_beams, timestep):
    '''Prepare dec_seq for decoder.

    Initialize with <S>

    Args:
        b_beams (beam, [batch_size,]) : Beam
        timestep: max_len(beam search len)

    Returns:
        dec_seq (torch.int64, [batch_size * beam_size, 1])
    '''
    dec_seq = [beam.get_hypothesis() for beam in b_beams if not beam.done]
    dec_seq = torch.stack(dec_seq).to(torch.device('cuda'))
    dec_seq = dec_seq.view(-1, timestep)
    return dec_seq


def prepare_beam_vis_key_padding_mask(b_beams, vis_key_padding_mask, beam_size):
    keep = []
    for idx, each in enumerate(vis_key_padding_mask):
        if not b_beams[idx].done:
            keep.append(idx)
    vis_key_padding_mask = vis_key_padding_mask[torch.tensor(keep)]
    len_s = vis_key_padding_mask.shape[-1]
    batch_size = vis_key_padding_mask.shape[0]
    vis_key_padding_mask = vis_key_padding_mask.repeat(1, beam_size).view(batch_size * beam_size, len_s)
    return vis_key_padding_mask


def predict_char(model, dec_seq, vis_feat, n_active_beams, beam_size, vis_key_padding_mask):
    sem_feat, sem_mask, sem_key_padding_mask = model.semantic_branch(dec_seq)
    pos_feat = model.positional_branch(sem_feat)
    dec_output = model.mdcdp(
        sem_feat,
        vis_feat,
        pos_feat,
        tgt_mask=sem_mask,
        tgt_key_padding_mask=sem_key_padding_mask,
        memory_key_padding_mask=vis_key_padding_mask,
    ).permute(1, 0, 2)          # B*beam_size, T, C
    dec_output = dec_output[:, -1, :]  # Pick the last step     # B*beam_size, C
    char_prob = model.tgt_word_prj(dec_output)                  # B*beam_size, K
    logit = char_prob.unsqueeze(0).view(n_active_beams, beam_size, -1)      # B, beam_size, K
    char_prob = F.log_softmax(char_prob, dim=1)
    char_prob = char_prob.view(n_active_beams, beam_size, -1)
    return char_prob, logit


def update_beams(b_beams, char_prob, logit, batch_idx_to_beam_idx):
    """Update beams & collect incomplete beam instances"""
    active_batch_indices = []
    for batch_idx, beam_idx in batch_idx_to_beam_idx.items():
        is_beam_complete = b_beams[batch_idx].update(char_prob[beam_idx], logit[beam_idx])
        if not is_beam_complete:
            active_batch_indices.append(batch_idx)
    return active_batch_indices


def collate_active_info(beam_size, enc_output, batch_idx_to_beam_idx, active_batch_indices):
    # Words which are still active are collected, so the decoder will not run on completed words.
    n_prev_active_inst = len(batch_idx_to_beam_idx)
    active_beam_indices = [batch_idx_to_beam_idx[b] for b in active_batch_indices]
    active_beam_indices = torch.LongTensor(active_beam_indices).to(torch.device('cuda'))
    active_enc_output = collect_active_part(enc_output.permute(1, 0, 2), active_beam_indices, n_prev_active_inst,
                                            beam_size).permute(1, 0, 2)
    active_batch_idx_to_beam_idx = get_batch_idx_to_beam_idx(active_batch_indices)
    return active_enc_output, active_batch_idx_to_beam_idx


def collect_active_part(beamed_tensor, curr_active_batch_idx, n_prev_active_inst, beam_size):
    ''' Collect tensor parts associated to active instances. '''
    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_batch_idx)
    new_shape = (n_curr_active_inst * beam_size, *d_hs)
    beamed_tensor = beamed_tensor.contiguous().view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_batch_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)
    return beamed_tensor


def collect_hypothesis_and_scores(b_beams):
    all_hyp, all_scores = [], []
    for batch_idx in range(len(b_beams)):
        scores, sort_indices = b_beams[batch_idx].sort_scores()
        all_scores += [scores[:1]]
        hyps = [b_beams[batch_idx].get_beam_hypothesis(i) for i in sort_indices[:1]]
        all_hyp += hyps

    return all_hyp, all_scores


def get_logit_from_beam(b_beams, batch_size, max_sequence_length=26, vocab_size=2354):
    logit = torch.zeros(max_sequence_length, batch_size, vocab_size, device=torch.device('cuda'))   # T_max, B, K

    # Allocate sufficiently large value for padding token
    logit[:, :, 0] = 10**6

    for batch_idx in range(len(b_beams)):
        feature_for_batch = b_beams[batch_idx].all_features
        scores, sort_indices = b_beams[batch_idx].sort_scores()
        node_idx = sort_indices[0]
        for j in reversed(range(len(b_beams[batch_idx].parent_nodes))):
            logit[j, batch_idx, :] = feature_for_batch[j][node_idx]
            node_idx = b_beams[batch_idx].parent_nodes[j][node_idx]

    return logit
