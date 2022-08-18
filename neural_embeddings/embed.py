import copy
import random
import unicodedata
from nltk.tokenize import sent_tokenize

import torch
from transformers import BertForMaskedLM, BertTokenizer
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from transformers import AdamW, get_constant_schedule


class EmbedNeural:
    """Creates neural embedding from text."""
    def __init__(
        self,
        model_name='bert-base-uncased',
        device='cpu',
        layers_emb = [
            'bert.encoder.layer.11.output.LayerNorm.weight',
            'bert.encoder.layer.11.output.LayerNorm.bias',
            'bert.encoder.layer.11.output.dense.bias'],
        mask_gaps = [3, 2, -3, -4],
        n_errors_per_epochs_max = -1,
        batch_size=30,
        input_tokens_max=510,
        n_epochs=10,
        learning_rate=5.0e-5,
        random_shuffle_inputs = True,
        random_seed=-1,
    ):
        """
        layers_emb (List[str]): names of layers to produce neural embeddings.
            Select only 1D layers layers_emb. (For simplicity not dealing here with 
            layers of hogher dim - they are too large for an embedding anyways.)
        mask_gaps (List[int]): simplified blueprints for masking. Examples:
            3: mask every 3d token
            2: mask every 2nd token
            -3: mask all except every 3d token
            -4: mask all except every 4th token
        n_errors_per_epochs_max (int): If >=0 then tuning is stop when number of errors
            in epoch is not higher than this value.
        """
        self.model_name = model_name
        self.device = device
        self.random_seed = random_seed 
        self.layers_emb = layers_emb
        self.mask_gaps = mask_gaps
        self.n_errors_per_epochs_max = n_errors_per_epochs_max
        self.batch_size = batch_size
        self.input_tokens_max = input_tokens_max
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_shuffle_inputs = random_shuffle_inputs 
        if model_name.find('distilbert') >= 0:
            self.model_tune = DistilBertForMaskedLM.from_pretrained(self.model_name)
            self.model_tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        else:
            self.model_tune = BertForMaskedLM.from_pretrained(self.model_name)
            self.model_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model_tune.eval()
        self.model_tune.to(device)
        self.emb_standard = self.get_embedding_standard() # original weights
        # convenient:
        self.id_label_ignore = -100
        self.inputs = None  # no masking, all inputs as token-ids

    def make_embedding(self, text):
        """
        Returns as torch vector in case further use or evaluation is in torch.
        Args:
            text (str)
        Returns:
            neural embedding as a normalized torch 1D vector.
        """
        if self.random_seed > 0:
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
        batches_ids, batches_att, batches_lbl = self._make_batches_from_text(text)
        self.model_tune.eval()
        self._reset_model()
        self.model_tune.train()
        self._prepare_layers_tune_freeze(self.model_tune)
        optimizer_parameters = [p for p in self.model_tune.parameters() if p.requires_grad]
        optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        scheduler = get_constant_schedule(optimizer)
        for _ in range(self.n_epochs):
            n_errors_in_epoch = 0
            for inp, att, lbl in zip(batches_ids, batches_att, batches_lbl):
                self.model_tune.zero_grad()
                optimizer.zero_grad()
                output = self.model_tune(
                    input_ids=inp.to(self.device),
                    attention_mask=att.to(self.device),
                    labels=lbl.to(self.device),
                )
                # Check prediction errors (if required):
                if self.n_errors_per_epochs_max >= 0:
                    for i_sample, sample_lbl in enumerate(lbl):
                        logits_of_sample = output['logits'][i_sample]
                        for ix_token, lbl_token in enumerate(sample_lbl):
                            if lbl_token != self.id_label_ignore:
                                logits_of_token = logits_of_sample[ix_token]
                                id_predicted = logits_of_token.argmax().item()
                                if id_predicted != lbl_token:
                                    n_errors_in_epoch += 1
                loss = output['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()
            if n_errors_in_epoch <= self.n_errors_per_epochs_max:
                break
        self.model_tune.eval()
        neural_emb = self._get_weights_emb(self.model_tune)
        neural_emb = [w - w0 for (w,w0) in zip(neural_emb, self.emb_standard)]
        neural_emb = [w / w.norm() for w in neural_emb]
        n_embs = len(neural_emb)
        neural_emb = torch.concat(neural_emb)
        if n_embs > 1:
            neural_emb = torch.nn.functional.normalize(neural_emb, dim=0)
        return neural_emb

    def get_embedding_standard(self):
        """Gets the original weights, for normalizing neural embeddings."""
        self.model_tune.eval()
        weights_of_layers = self._get_weights_emb(self.model_tune)
        weights_emb = []
        for weights in weights_of_layers:
            weights_emb.append(weights)
        self.model_tune.train()
        return weights_emb

    def _reset_model(self):
        """Only 1D layers:"""
        for i_layer, layer in enumerate(self.layers_emb):
            w = self.model_tune.state_dict()[layer]
            w[:] = self.emb_standard[i_layer]  # 1D layers only

    def _prepare_layers_tune_freeze(self, model):
        for name, param in model.named_parameters():
            if name in self.layers_emb:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _get_weights_emb(self, model):
        weights_emb = []
        for layer_name in self.layers_emb:
            weights = copy.deepcopy(model.state_dict()[layer_name])
            weights_emb.append(weights.flatten())
        return weights_emb

    def _make_batches_from_text(self, text):
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
        if self.random_shuffle_inputs:
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