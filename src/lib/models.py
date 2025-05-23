import math

from .params import *

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Taken from:
        https://github.com/pytorch/examples/blob/main/word_language_model/model.py

        With some changes because theirs didn't work.
    """
    def __init__(self, d_model: int, dropout=0.1, max_len=2600):
        super(PositionalEncoding, self).__init__()

        # Dropout incredbily slow?
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.pe = torch.zeros(max_len, d_model).to(device)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)[:, (0 if d_model % 2 == 0 else 1):]
        # self.pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        # self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', self.pe)


    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape `[batch, seq_len, embedding_dim]`
        """
        # x = (x + self.pe[:x.size(0), :]).unsqueeze(1)
        # Add batch dimension
        x = self.pe[None, :x.size(1), :] + x
        return self.dropout(x)
        # return x # No dropout



class BaseTransformer(nn.Module):
    def __init__(self, d_model, num_head=2, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=512,
                 dropout=0.1, layer_norm_eps=1e-5, batch_first=True, name="Test_1", activation="relu", make_mask=True, dropout_pe=0.1):
        super().__init__()

        self.name = name

        self.pos_encoding = PositionalEncoding(d_model, dropout_pe)

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_head, dim_feedforward, dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_head, dim_feedforward, dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.src_mask = None
        self.make_mask = make_mask


    def forward(self, x, make_src_mask=True, **kwargs):
        if make_src_mask and self.make_mask:
            if self.src_mask is None or self.src_mask.size(0) != x.shape[1]:
                mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = x
        x = self.pos_encoding(x)
        output = self.encoder(x, mask=self.src_mask, **kwargs)
        output = self.decoder(src, output, **kwargs)
        return output

    def get_name(self):
        return self.name


class IndBasedEmbedding(nn.Module):
    def __init__(self, seqlen, embed_dim):
        """Dumb embedding, where the token indices are just the indices of each row in the given input.

        Args:
            seqlen (int): Max length of input sequence
            embed_dim (int): The size of the output embedding dimension
        """
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=seqlen, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor):
        """Assumes x dim = (b, n, m) where b is batch size, n is seqlength, m is existing embedding dim.
        Returns of size (b, n, m + embed_dim)

        Args:
            x (tensor): input
        """
        inds = torch.tensor(range(x.shape[1]), dtype=int).to(device).expand(x.shape[0], -1)
        out = self.embed(inds)
        return torch.cat((x, out), dim=2)



class EmbedTransformer(BaseTransformer):
    """Same as BaseTransformer, except that potentially no positional encoding and potentially an additional
    learned embedding layer is present. The NN will use therefore d_model + embed_layer as the final embed dim size.
    The embedding layer is dumb and simple, uses the row inds as the tokens.

    Args:
        d_model (int): The size of the embedding dim in the input
        num_head (int): The number of self-attention heads in an attention layer
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        name (str): The assigned name to this model, for debugging purposes
        dim_feedforward (int, optional): The dim of the linear layers in each encoder/decoder layer.
            Defaults to 512.
        dropout (float, optional): Dropout frac. Defaults to 0.1.
        layer_norm_eps (float, optional): Layer norming ?? Defaults to 1e-5.
        activation (str | function, optional): Activation after each layer. Defaults to "relu".
        dropout_pe (float, optional): The dropout frac in positional encoding. Defaults to 0.1.
        use_pe (bool, optional): Whether or not to use positional encoding. Defaults to True.
        embed_layer (int, optional): If nonzero, then an embedding layer of dict size seq_length with
            embedding size embed_layer; else no embedding layer. Defaults to 0.
        seq_length (int, optional): Max length of seq, also the actual fixed length of seq in our case.
            Defaults to 2503.
    """
    def __init__(self, d_in, num_head, num_encoder_layers,
                 num_decoder_layers, name, dim_feedforward=512,
                 dropout=0.1, layer_norm_eps=1e-5, activation="relu", dropout_pe=0.1,
                 use_pe=True, embed_layer=0, seq_length=2503):

        # Just always make batch first dim
        batch_first = True
        # No causal masking
        make_mask=False

        # If adding the embedding, now the d_model is bigger
        if embed_layer >= 0:
            d_model = d_in + embed_layer
        else:
            raise Exception("No embed_layer < 0 or not int")

        super().__init__(d_model, num_head, num_encoder_layers,
                        num_decoder_layers, dim_feedforward,
                        dropout, layer_norm_eps, batch_first,
                        name, activation, make_mask, dropout_pe)

        if not use_pe:
            self.pos_encoding = nn.Identity()

        if embed_layer > 0:
            self.embed = IndBasedEmbedding(seq_length, embed_layer)
        else:
            self.embed = nn.Identity()

        # Another linear layer never hurt anyone
        self.outshape = nn.Linear(in_features=d_model, out_features=d_in)


    def forward(self, x, **kwargs):
        src = self.embed(x)
        x = self.pos_encoding(src)
        output_encoder = self.encoder(x, mask=None, **kwargs)
        output = self.outshape(self.decoder(src, output_encoder, **kwargs))
        return output



class TokenizedInputTransformer(BaseTransformer):
    def __init__(self, d_model, num_head, num_encoder_layers,
                    num_decoder_layers, name, dim_feedforward=512,
                    dropout=0.1, layer_norm_eps=1e-5, activation="relu", dropout_pe=0.1,
                    use_pe=True, maxseqlen=2502, maxind=2502):

        # Just always make batch first dim
        batch_first = True
        # No causal masking
        make_mask=False

        # The size seen by model is the embedding dim size (d_model = embed_size)

        super().__init__(d_model, num_head, num_encoder_layers,
                        num_decoder_layers, dim_feedforward,
                        dropout, layer_norm_eps, batch_first,
                        name, activation, make_mask, dropout_pe)

        if not use_pe:
            self.pos_encoding = nn.Identity()

        # Num_embeddings must be len of all inds (+1), along with the extra
        # padding index at the end (+1)
        self.padind = maxind+1
        self.embed = nn.Embedding(num_embeddings=maxind+2,
                                  embedding_dim=d_model,
                                  padding_idx=self.padind)


        # Linear layer for right shape output
        self.outshape = nn.Linear(in_features=d_model, out_features=maxseqlen)


    def forward(self, x, key_padding_mask, **kwargs) -> torch.Tensor:
        src = self.embed(x)
        x = self.pos_encoding(src)
        output_encoder = self.encoder(x,
                                      src_key_padding_mask=key_padding_mask,
                                      **kwargs)
        output_decoder = self.decoder(src,
                                      output_encoder,
                                      tgt_key_padding_mask=key_padding_mask,
                                      memory_key_padding_mask=key_padding_mask,
                                      **kwargs)
        output = self.outshape(output_decoder)
        return output


class TokenizedPopTransformer(TokenizedInputTransformer):
    def __init__(self, d_model, num_head, num_encoder_layers,
                num_decoder_layers, name, num_classes, dim_feedforward=512,
                dropout=0.1, layer_norm_eps=1e-5, activation="relu", dropout_pe=0.1,
                use_pe=True, maxseqlen=2502, maxind=2502):

        super().__init__(d_model, num_head, num_encoder_layers,
                num_decoder_layers, name, dim_feedforward,
                dropout, layer_norm_eps, activation, dropout_pe,
                use_pe, maxseqlen, maxind)

        self.outshape = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x, key_padding_mask, **kwargs):
        x = super().forward(x, key_padding_mask, **kwargs)
        return x.transpose(1, 2)
