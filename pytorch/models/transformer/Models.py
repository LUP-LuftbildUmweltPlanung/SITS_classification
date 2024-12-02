''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
#import pytorch.models.transformer.Constants as Constants
from pytorch.models.transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"
Pad_Value = 0
def get_non_pad_mask(seq):
    return seq[:,:,0].ne(Pad_Value).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Pad_Value)
    padding_mask = padding_mask[:,:,0].unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
def positional_encoding(positions, d_model, max_seq_length, pad_value=Pad_Value):
    ''' Compute positional encodings for arbitrary positions. '''
    # positions: tensor of shape (batch_size, seq_len)
    # returns: tensor of shape (batch_size, seq_len, d_model)
    #print(max_seq_length)
    #scale_factor = max_seq_length / 10000
    #positions = positions.float() / scale_factor
    #max_seq_length = 10000
    
    angle_rates = 1 / torch.pow(max_seq_length, (2 * (torch.arange(d_model) // 2).float()) / d_model)
    if positions.is_cuda:
        angle_rates = angle_rates.cuda()
    angle_rads = positions.unsqueeze(-1).float() * angle_rates  # (batch_size, seq_len, d_model)

    #print(angle_rads.shape)
    # Apply sin to even indices and cos to odd indices
    pos_encoding = torch.zeros_like(angle_rads)
    pos_encoding[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
    pos_encoding[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])
    # Zero out positional encodings at padding positions
    mask = positions.eq(pad_value).unsqueeze(-1)
    pos_encoding = pos_encoding.masked_fill(mask, 0)
    return pos_encoding

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.n_position = len_max_seq
        #self.src_word_emb = nn.Embedding(
        #   n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.d_model = d_model

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, src_pos_month, src_thermal, mask_x, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=mask_x, seq_q=mask_x)
        #slf_attn_mask = torch.zeros((src_seq.shape[0],src_seq.shape[1],src_seq.shape[1]),dtype=torch.uint8)
        non_pad_mask = get_non_pad_mask(mask_x)

        if torch.cuda.is_available():
            slf_attn_mask = slf_attn_mask.cuda()
            non_pad_mask = non_pad_mask.cuda()

        masked_src_seq = src_seq * non_pad_mask.float()  # Convert mask to float for multiplication
        # Apply mask to src_pos to ignore padded positions in positional encoding
        masked_src_pos = src_pos * non_pad_mask.squeeze(-1).long()  # Assuming non_pad_mask is broadcastable to src_pos dimensions

        if src_thermal is not None:
            masked_src_thermal = (src_thermal * non_pad_mask.squeeze(-1)).long()
            #print(masked_src_thermal[0,:])
            thermal_pos_encodings = positional_encoding(masked_src_thermal, self.d_model, max_seq_length=10000)
            doy_pos_encodings = positional_encoding(masked_src_pos, self.d_model, max_seq_length=self.n_position)
            #print(thermal_pos_encodings[0,:,0])
            #print(doy_pos_encodings[0, :, 0])
            enc_output = masked_src_seq + thermal_pos_encodings + doy_pos_encodings

        else:
            # -- Forward self.src_word_emb(src_seq)
            masked_src_pos_month = src_pos_month * non_pad_mask.squeeze(-1)  # Assuming non_pad_mask is broadcastable to src_pos dimensions
            masked_src_pos_month = masked_src_pos_month.long()
            doy_pos_encodings = positional_encoding(masked_src_pos, self.d_model, max_seq_length=self.n_position)
            month_pos_encodings = positional_encoding(masked_src_pos_month, self.d_model, max_seq_length=13)
            enc_output = masked_src_seq + doy_pos_encodings + month_pos_encodings


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
