import copy
import random
import torch
from transformers import BertTokenizer, AutoConfig, BertForMaskedLM, BertModel
from torch.optim import AdamW


class EmbedNeural:
    """Creates neural embedding from text."""
    def __init__(
        self,
        preparator_inputs,  # instance of PrepInputs
        model_name='bert-base-uncased',
        device='cpu',
        layers_emb = [
            'cls.predictions.transf orm.dense.bias',
            'cls.predictions.transform.LayerNorm.weight',
            'cls.predictions.transform.LayerNorm.bias'],
        id_block_frozen_top = 11,
        keep_all_batches_on_device=True,
        keep_hidden_states=False,
        random_shuffle_batches = False,
        n_epochs=10,
        n_tuneups=1,
        learning_rate=1.0e-2,
        random_seed=1,
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
        """
        self.model_name = model_name
        self.device = device
        self.random_seed = random_seed 
        self.layers_emb = layers_emb
        self.id_block_frozen_top = id_block_frozen_top
        self.keep_all_batches_on_device = keep_all_batches_on_device
        self.n_epochs = n_epochs
        self.n_tuneups = n_tuneups
        self.keep_hidden_states = keep_hidden_states
        if self.n_epochs == 1:
            self.keep_hidden_states = False
        if self.n_tuneups == 1 and self.keep_hidden_states == False:
            self.use_hidden_states = False
        else:
            self.use_hidden_states = True
        self.learning_rate = learning_rate
        self.random_shuffle_batches = random_shuffle_batches 
        self.model_tune = BertForMaskedLM.from_pretrained(self.model_name, output_hidden_states=self.use_hidden_states)
        self.model_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.vocab_size = AutoConfig.from_pretrained(self.model_name).vocab_size
        self.model_tune.eval()
        self.model_tune.to(device)
        optimizer_parameters = []
        self.dim_emb = 0  # convenient, for info
        for name, param in self.model_tune.named_parameters():
            if name in self.layers_emb:
                param.requires_grad = True
                optimizer_parameters.append(param)
                self.dim_emb += len(param.flatten())
            else:
                param.requires_grad = False 
        self.optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        self.model_tune.eval()
        self.emb_standard = get_weights_of_layers(self.model_tune, self.layers_emb)
        # convenient:
        self.preparator_inputs = preparator_inputs
        self.id_label_ignore = -100
        self.inputs = None
        self.loss_ce = torch.nn.CrossEntropyLoss()

    def __call__(self, texts):
        return [self.make_embedding(t) for t in texts]

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
        batches_ids, batches_att, batches_lbl = self.preparator_inputs(text)
        n_batches = len(batches_ids)
        if self.random_shuffle_batches and n_batches > 1:
            ixs = list(range(n_batches))
        self._reset_model()
        self.model_tune.train()
        hidden_states_batches = []
        for i_epoch in range(self.n_epochs):
            if i_epoch > 0 and self.random_shuffle_batches and n_batches > 1:
                random.shuffle(ixs)
                batches_ids = [batches_ids[i] for i in ixs]
                batches_att = [batches_att[i] for i in ixs]
                batches_lbl = [batches_lbl[i] for i in ixs]
                if hidden_states_batches:
                    hidden_states_batches = [hidden_states_batches[i] for i in ixs]
            for i_batch, (inp, att, lbl) in enumerate(zip(batches_ids, batches_att, batches_lbl)):
                if not self.keep_all_batches_on_device:  # output of preparator_inputs is not on device
                    inp = inp.to(self.device)
                    att = att.to(self.device)
                    lbl = lbl.to(self.device)
                hidden_state_tuneup = None
                for i_tuneup in range(self.n_tuneups):
                    self.model_tune.zero_grad()
                    self.optimizer.zero_grad()
                    if self.keep_hidden_states:
                        if i_epoch == 0 and i_tuneup == 0:
                            output = self.model_tune(input_ids=inp, attention_mask=att, labels=lbl)
                            loss = output['loss']
                            hidden_states_batches.append(output.hidden_states[self.id_block_frozen_top])
                        else:
                            loss = self._get_loss_from_hidden(hidden_states_batches[i_batch], lbl)
                    else:
                        if i_tuneup == 0:
                            output = self.model_tune(input_ids=inp, attention_mask=att, labels=lbl)
                            loss = output['loss']
                            if self.n_tuneups > 1:
                                hidden_state_tuneup = output.hidden_states[self.id_block_frozen_top]
                        else:
                            loss = self._get_loss_from_hidden(hidden_state_tuneup, lbl)
                    loss.backward()
                    self.optimizer.step()
        self.model_tune.eval()
        neural_emb = get_weights_of_layers(self.model_tune, self.layers_emb)
        neural_emb = [w - w0 for (w,w0) in zip(neural_emb, self.emb_standard)]
        neural_emb = [w / w.norm() for w in neural_emb]
        n_embs = len(neural_emb)
        neural_emb = torch.concat(neural_emb)
        if n_embs > 1:
            neural_emb = torch.nn.functional.normalize(neural_emb, dim=0)
        return neural_emb

    def _reset_model(self):
        """Only 1D layers:"""
        for i_layer, layer in enumerate(self.layers_emb):
            w = self.model_tune.state_dict()[layer]
            w[:] = self.emb_standard[i_layer]
        optimizer_parameters = [p for p in self.model_tune.parameters() if p.requires_grad]
        self.optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)

    def _get_loss_from_hidden(self, hidden_state, labels):
        out = self.model_tune.bert.encoder.layer[self.id_block_frozen_top].forward(hidden_state)
        out_cls = self.model_tune.cls.forward(out[0])
        loss = self.loss_ce(out_cls.view(-1, self.vocab_size), labels.view(-1))
        return loss


class EmbedNeuralTop:
    """Creates neural embedding from text.
    Simplified and fast version. Select embedding layers only from the top (cls) part.
    Base model part provides hidden states at first epoch; 
    top part micro-tunes at all epochs.
    """
    def __init__(
        self,
        preparator_inputs,
        model_name='bert-base-uncased',
        device='cpu',
        layers_emb = [
            'predictions.transform.dense.bias',
            'predictions.transform.LayerNorm.weight', 
            'predictions.transform.LayerNorm.bias'],
        keep_all_batches_on_device=True,
        keep_hidden_states=True,
        random_shuffle_batches=True,
        n_epochs=30,
        learning_rate=5.0e-5,
        dropout=-1,
        random_seed=1
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
        """
        self.model_name = model_name
        self.device = device
        self.random_seed = random_seed 
        self.random_shuffle_batches = random_shuffle_batches
        self.layers_emb = layers_emb
        self.keep_all_batches_on_device = keep_all_batches_on_device
        self.n_epochs = n_epochs
        self.keep_hidden_states = keep_hidden_states
        if self.n_epochs == 1:
            self.keep_hidden_states = False
        self.learning_rate = learning_rate
        self.model_base = BertModel.from_pretrained(self.model_name)
        self.model_base.eval()
        for p in self.model_base.parameters():
            p.requires_grad=False
        self.model_base.to(self.device)
        model_tune = BertForMaskedLM.from_pretrained(self.model_name)
        self.model_top = model_tune.cls
        self.model_top.to(self.device)
        self.dropout = None
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout).to(self.device)
        optimizer_parameters = []
        self.dim_emb = 0  # convenient, for info
        for name, param in self.model_top.named_parameters():
            if name in self.layers_emb:
                param.requires_grad = True
                optimizer_parameters.append(param)
                self.dim_emb += len(param.flatten())
            else:
                param.requires_grad = False 
        self.optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        self.model_top.eval()
        self.emb_standard = get_weights_of_layers(self.model_top, self.layers_emb) # original weights
        self.preparator_inputs = preparator_inputs
        # convenient:
        self.vocab_size = AutoConfig.from_pretrained(self.model_name).vocab_size
        self.loss_ce = torch.nn.CrossEntropyLoss()

    def __call__(self, texts):
        return [self.make_embedding(t) for t in texts]

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
        batches_ids, batches_att, batches_lbl = self.preparator_inputs(text)
        n_batches = len(batches_ids)
        if self.random_shuffle_batches and n_batches > 1:
            ixs = list(range(n_batches))
        self._reset_model()
        self.model_top.train()
        hidden_states_batches = []
        for i_epoch in range(self.n_epochs):
            if i_epoch > 0 and self.random_shuffle_batches and n_batches > 1:
                random.shuffle(ixs)
                batches_ids = [batches_ids[i] for i in ixs]
                batches_att = [batches_att[i] for i in ixs]
                batches_lbl = [batches_lbl[i] for i in ixs]
                if hidden_states_batches:
                    hidden_states_batches = [hidden_states_batches[i] for i in ixs]
            for i_batch, (inp, att, lbl) in enumerate(zip(batches_ids, batches_att, batches_lbl)):
                if not self.keep_all_batches_on_device:  # output of preparator_inputs is not on device
                    inp = inp.to(self.device)
                    att = att.to(self.device)
                    lbl = lbl.to(self.device)
                if i_epoch == 0 or not self.keep_hidden_states:
                    hidden_state = self.model_base(input_ids=inp, attention_mask=att).last_hidden_state
                    if self.keep_hidden_states:
                        hidden_states_batches.append(hidden_state)
                if i_epoch > 0 and self.keep_hidden_states:
                    hidden_state = hidden_states_batches[i_batch]
                self.model_top.zero_grad()
                self.optimizer.zero_grad()
                if self.dropout:
                    hidden_state = self.dropout(hidden_state)
                out = self.model_top(hidden_state)
                masked_lm_loss = self.loss_ce(out.view(-1, self.vocab_size), lbl.view(-1))
                masked_lm_loss.backward()
                self.optimizer.step()
        self.model_top.eval()
        neural_emb = get_weights_of_layers(self.model_top, self.layers_emb)
        neural_emb = [w - w0 for (w,w0) in zip(neural_emb, self.emb_standard)]
        neural_emb = [w / w.norm() for w in neural_emb]
        neural_emb = torch.concat(neural_emb)
        if len(self.layers_emb) > 1:
            neural_emb = torch.nn.functional.normalize(neural_emb, dim=0)
        return neural_emb

    def _reset_model(self):
        """Only 1D layers:"""
        for i_layer, layer in enumerate(self.layers_emb):
            w = self.model_top.state_dict()[layer]
            w[:] = self.emb_standard[i_layer]  # 1D layers only
        optimizer_parameters = [p for p in self.model_top.parameters() if p.requires_grad]
        self.optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)


def get_weights_of_layers(model,layers):
    weights = []
    for layer_name in layers:
        weights_layer = copy.deepcopy(model.state_dict()[layer_name])
        weights.append(weights_layer.flatten())
    return weights