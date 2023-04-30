import torch
import torch.nn as nn
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from torch import Tensor
from utils import PositionalEncoder, ScalPositionalEncoding, PAD_IDX

class Transformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoder(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output

class MelodyTransformer(nn.Module):
    def __init__(self, encoder_args, decoder_args, 
                 src_vocab_size, tgt_vocab_size, dropout = 0.1):
        super().__init__()
        self.encoder_type = 'Transformer'
        self.decoder_type = 'Transformer'
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        
        # Encoder same as base chord transformer without linear output layer
        self.pos_encoder = PositionalEncoder(encoder_args.d_model, dropout)
        encoder_layers = TransformerEncoderLayer(encoder_args.d_model,encoder_args.nhead, encoder_args.d_hid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, encoder_args.nlayers)

        decoder_layers = TransformerDecoderLayer(decoder_args.d_model, decoder_args.nhead, encoder_args.d_hid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, decoder_args.nlayers)

        # d_model is embedding size
        self.generator = nn.Linear(decoder_args.d_model, tgt_vocab_size)
        self.src_emb = nn.Embedding(src_vocab_size, encoder_args.d_model, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, decoder_args.d_model)

        self.init__weights()
    
    def init_weights(self): 
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, tgt, tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.src_emb(src) * math.sqrt(self.encoder_args.d_model)
        tgt = self.tgt_emb(src) * math.sqrt(self.decoder_args.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, output, tgt_mask=tgt_mask, memory_mask=src_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        output = self.generator(output)
        return output

