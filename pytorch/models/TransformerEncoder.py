import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from pytorch.models.ClassificationModel import ClassificationModel
from pytorch.models.transformer.Models import Encoder
from datetime import datetime, timedelta
import numpy as np

class TransformerEncoder(ClassificationModel):
    def __init__(self, in_channels=13, len_max_seq=100,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.2, nclasses=6, response=None):

        self.response = response
        self.d_model = d_model
        super(TransformerEncoder, self).__init__()

        self.inlayernorm = nn.LayerNorm(in_channels)
        self.convlayernorm = nn.LayerNorm(d_model)
        self.outlayernorm = nn.LayerNorm(d_model)

        self.inconv = torch.nn.Conv1d(in_channels, d_model, 1)
        #self.inconv_bn = nn.BatchNorm1d(d_model)

        self.encoder = Encoder(
            n_src_vocab=None, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.outlinear = nn.Linear(d_model, nclasses, bias=False)

        self.tempmaxpool = nn.AdaptiveMaxPool1d(1)
        #self.tempmaxpool = nn.MaxPool1d(int(len_max_seq))


    def _logits(self, x, doy, thermal = None):

        # b,d,t - > b,t,d
        x = x.transpose(1,2)

        mask_x = x

        x = self.inlayernorm(x)
        x = self.inconv(x.transpose(1,2)).transpose(1,2)
        #x = self.inconv_bn(x)
        x = self.convlayernorm(x)

        #batchsize, seq, d = x.shape
        #print(seq)
        #src_pos = torch.arange(1, seq + 1, dtype=torch.long).expand(batchsize, seq)
        src_pos = doy.long()
        if thermal is not None:
            thermal = thermal.long()
            thermal = thermal.cuda()
            src_pos_month = None

        else:
            doy_y = torch.remainder(doy, 365)  #
            src_pos_month = doy_y.long()
            src_pos_month = src_pos_month.cuda()

        src_pos = src_pos.cuda()

        enc_output, enc_slf_attn_list = self.encoder.forward(src_seq=x, src_pos=src_pos, src_pos_month=src_pos_month, src_thermal = thermal, mask_x = mask_x, return_attns=True)
        enc_output = self.outlayernorm(enc_output)
        ##########masking for padding
        mask = mask_x.sum(dim=-1) > 0
        mask_unsqueezed = mask.unsqueeze(-1)
        enc_output = enc_output * mask_unsqueezed.float()
        ##########masking for padding ende

        enc_output = self.tempmaxpool(enc_output.transpose(1, 2)).squeeze(-1)

        logits = self.outlinear(enc_output)


        return logits, None, None, None

    def forward(self, x, doy, thermal = None):
        logits, *_ = self._logits(x, doy, thermal)

        if self.response == "classification":
            logprobabilities = F.log_softmax(logits, dim=-1)
        elif self.response == "regression_relu":
            logprobabilities = F.relu(logits)
        elif self.response == "regression_sigmoid":
            logprobabilities = torch.sigmoid(logits)
        elif self.response == "regression":
            logprobabilities = logits  # Return logits directly without any activation
        else:
            # If response is not one of the predefined types, raise an error
            raise ValueError("Response type must be 'classification', 'regression', 'regression_relu', or 'regression_sigmoid'.")

        return logprobabilities, None, None, None

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

