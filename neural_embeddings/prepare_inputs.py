import copy
import random
import unicodedata
from nltk.tokenize import sent_tokenize
import torch


class PrepInputs:
    """Creates inputs for micro-tuning."""
    def __init__(
        self,
        model_tokenizer,
        device='cpu',
        mask_gaps = [3, 2, -3, -4],
        keep_all_batches_on_device=False,
        batch_size=10,
        input_tokens_max=510,
        random_shuffle_inputs = True,
    ):
        """
        mask_gaps (List[int]): simplified blueprints for masking. Examples:
            3: mask every 3d token
            2: mask every 2nd token
            -3: mask all except every 3d token
            -4: mask all except every 4th token
        keep_all_batches_on_device (Boolean): If device is 'cuda' (gpu), 
            setting this to True makes faster processing. Since the size of 
            the 'dataset' (all batches) in micro-tuning is extremely small,
            it should be almost always possible to set this to True.
        """
        self.model_tokenizer = model_tokenizer
        self.device = device
        self.mask_gaps = mask_gaps
        self.keep_all_batches_on_device = keep_all_batches_on_device
        self.batch_size = batch_size
        self.input_tokens_max = input_tokens_max
        self.random_shuffle_inputs = random_shuffle_inputs
        # convenient:
        self.id_label_ignore = -100
        self.inputs = None
        

    def __call__(self, text):
        return self.make_batches_from_text(text)

    def make_batches_from_text(self, text):
        self._prepare_inputs(text)
        inputs_for_batches = self._prepare_inputs_for_batches()
        batches_ids, batches_att, batches_lbl = self._make_batches(inputs_for_batches)
        return batches_ids, batches_att, batches_lbl

    def _prepare_inputs(self, text):
        """Sets list of inputs, each input is list of token ids."""
        chunks_tokens_ids = tokenize_text(
            text, 
            tokenizer=self.model_tokenizer, 
            n_tokens_max=self.input_tokens_max)
        self.inputs = []  # all inputs
        for chunk_tokens_ids in chunks_tokens_ids:
            input = [self.model_tokenizer.cls_token_id] + chunk_tokens_ids + [self.model_tokenizer.sep_token_id]
            self.inputs.append(input) 

    def _prepare_inputs_for_batches(self):
        """
        Returns:
            inputs_for_batches (List[(List[int],List[int])]): List of duples,
                each duple is input token ids and labels.
        """
        inputs_for_batches = []
        for ids_input in self.inputs:  # through batches
            for mask_gap in self.mask_gaps:
                (mask_gap, opposite) = (mask_gap, False) if mask_gap>0 else (-mask_gap, True)
                mask_shift_end = min(mask_gap, len(ids_input)) + 1
                for mask_shift in range(mask_shift_end):
                    ids = copy.deepcopy(ids_input)
                    lbl = [self.id_label_ignore] * len(ids)
                    have_a_mask = False
                    for ix in range(1, len(ids)):
                        marked = (ix-mask_shift)%mask_gap == 0
                        if ix != len(ids)-1 and opposite != marked:
                            have_a_mask = True
                            lbl[ix] = ids[ix]
                            ids[ix] = self.model_tokenizer.mask_token_id
                    if have_a_mask:
                        inputs_for_batches.append((ids,lbl))
        return inputs_for_batches

    def _make_batches(self, inputs_for_batches):
        if self.random_shuffle_inputs and len(inputs_for_batches) >= self.batch_size:
            random.shuffle(inputs_for_batches)
        batches_ids, batches_lbl, batches_att = [], [], []
        batch_ids_raw, batch_lbl_raw = [], []
        for ids, lbl in inputs_for_batches:
            batch_ids_raw.append(ids)
            batch_lbl_raw.append(lbl)
            if len(batch_ids_raw) == self.batch_size:
                batch_ids, batch_lbl, batch_att = self._square_batch_samples(batch_ids_raw, batch_lbl_raw)
                batches_ids.append(batch_ids)
                batches_lbl.append(batch_lbl)
                batches_att.append(batch_att)
                batch_ids_raw, batch_lbl_raw = [], []
        if len(batch_ids_raw) > 1:
            batch_ids, batch_lbl, batch_att = self._square_batch_samples(batch_ids_raw, batch_lbl_raw)
            batches_ids.append(batch_ids)
            batches_lbl.append(batch_lbl)
            batches_att.append(batch_att)
        return batches_ids, batches_att, batches_lbl

    def _square_batch_samples(self, batch_ids_raw, batch_lbl_raw):
        """Make all samples the same length"""
        len_max = max([len(ids) for ids in batch_ids_raw])
        batch_ids, batch_lbl, batch_att = [], [], []
        for ids, lbl in zip(batch_ids_raw, batch_lbl_raw):
            att = [1] * len(ids)
            while len(ids) < len_max:
                ids.append(0)
                lbl.append(self.id_label_ignore)
                att.append(0)
            batch_ids.append(ids)
            batch_lbl.append(lbl)
            batch_att.append(att)
        if self.keep_all_batches_on_device:
            batch_ids = torch.tensor(batch_ids).to(self.device)
            batch_att = torch.tensor(batch_att).to(self.device)
            batch_lbl = torch.tensor(batch_lbl).to(self.device)
        else:
            batch_ids = torch.tensor(batch_ids)
            batch_att = torch.tensor(batch_att)
            batch_lbl = torch.tensor(batch_lbl)
        return batch_ids, batch_lbl, batch_att


def tokenize_text(text, tokenizer, n_tokens_max=510):
    """
    Splits the text on inputs, filling each input with as many sentences as fits in.
    Keeps if possible whole sentences (unless some sentence is extremely long).
    Args:
        text (str)
        tokenizer: model tokenizer
    Returns:
        inputs_tokens_ids (List[List[int]]): List of inputs, each input is a list of token ids.
    """
    text = unicodedata.normalize('NFKD', text)
    text_sents = sent_tokenize(text)
    text_sents_tokens = [tokenizer.tokenize(sent) for sent in text_sents]
    text_sents_tokens = split_long_sentences(text_sents_tokens, n_tokens_max)
    text_sents_tokens_cut, n_toks_tot = [], 0
    for sent_tokens in text_sents_tokens:
        n_toks_tot += len(sent_tokens)
        text_sents_tokens_cut.append(sent_tokens)
    inputs_tokens_ids, input_tokens, n_tokens = [], [], 0
    for sent_tokens in text_sents_tokens_cut:
        n_sent_tokens = len(sent_tokens)
        if n_tokens + n_sent_tokens > n_tokens_max:
            if input_tokens:
                input_tokens_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                inputs_tokens_ids.append(input_tokens_ids)
                input_tokens = []
                n_tokens = 0
        input_tokens.extend(sent_tokens)
        n_tokens += len(sent_tokens)
    if input_tokens:
        input_tokens_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        inputs_tokens_ids.append(input_tokens_ids)
    return inputs_tokens_ids


def split_long_sentences(sents_tokenized, n_tokens_max=510):
    """
    Args:
        sents_tokenized (List[List[str]]): List of sentences,
            each sentence is a list of tokens.
    """
    sents_short_tokenized = []
    for sent in sents_tokenized:
        n_parts = (len(sent)-1) // n_tokens_max + 1
        if n_parts <= 1:
            sents_short_tokenized.append(sent)
        else:
            sents_short = [sent[i:i + n_tokens_max] for i in range(0, len(sent), n_tokens_max)]
            sents_short_tokenized.extend(sents_short)
    return sents_short_tokenized