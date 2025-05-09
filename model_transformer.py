import math
import torch
import time
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
  def __init__(self, d_dim, max_len=5000):
    super().__init__()

    # Throw an error if d_dim is odd
    if d_dim % 2 != 0:
      raise ValueError("d_dim must be even.")
    
    pe = torch.zeros(max_len, d_dim)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_dim, 2).float() * (-math.log(10000.0) / d_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    
    pe = pe.unsqueeze(0)  # shape (1, max_len, d_dim)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x shape: (batch_size, sequence_length, d_dim)
    x = x + self.pe[:, :x.size(1)]
    return x


class AttentionPooling(nn.Module):
  def __init__(self, d_model):
    super().__init__()

    self.query = nn.Parameter(torch.randn(d_model))

    # self.attn_layer = nn.Sequential(
    #   nn.Linear(d_model, d_model),
    #   nn.Tanh(),
    #   nn.Linear(d_model, 1),
    # )

  def forward(self, H):
    # H is transformer output of shape = (batch_size, seq_length, d_model)

    # Compute attention scores (with dot product)
    scores = H @ self.query

    # Compute attention scores (with MLP)
    # scores = self.attn_layer(H).squeeze(-1)
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=1)

    # Weighted sum of H (batch_size, d_model)
    summary = torch.sum(H * attn_weights.unsqueeze(-1), dim=1)

    return summary

class Encoder(nn.Module):
  def __init__(self, trace_attributes, num_activities, num_resources, max_trace_length,
               d_dim, transformer_nhead, transformer_dim_feedforward, transformer_num_layers,
               attr_e_dim, act_e_dim, res_e_dim, z_dim, c_dim, dropout_p,
               ):
    super(Encoder, self).__init__()

    self.embed_acts = nn.Embedding(
      num_embeddings=num_activities+1,
      embedding_dim=act_e_dim,
      padding_idx=num_activities,
    )
    self.padding_idx = num_activities

    self.pe = PositionalEncoding(d_dim)

    self.transformer_encoder = nn.TransformerEncoder(
      encoder_layer=nn.TransformerEncoderLayer(
        d_model=d_dim,
        nhead=transformer_nhead,
        dim_feedforward=transformer_dim_feedforward,
        dropout=dropout_p,
        batch_first=True,
      ),
      num_layers=transformer_num_layers,
    )

    # self.conv = nn.Conv1d(max_trace_length, 1, 1)
    self.attn_pool = AttentionPooling(d_dim)
    # self.lstm = nn.LSTM(
    #   input_size=d_dim,
    #   hidden_size=200,
    #   num_layers=1,
    #   batch_first=True,
    # )


    self.linear_mean = nn.Linear(d_dim+c_dim, z_dim)
    self.linear_var = nn.Linear(d_dim+c_dim, z_dim)

    # in case of LSTM
    # self.linear_mean = nn.Linear(200+c_dim, z_dim)
    # self.linear_var = nn.Linear(200+c_dim, z_dim)

  def forward(self, x, c=None):
    attrs, acts, ts, ress = x

    # build padding mask for transformer encoder (BS, TL)
    padding_mask_encoder = (acts == self.padding_idx)

    # embed acts (BS, TL, ACT_E_DIM)
    acts_embedded = self.embed_acts(acts)

    # concat acts, ress, ts, ...
    # ... todo ...
    e = acts_embedded

    # apply positional encoding (BS, TL, E_DIM)
    e_embedded = self.pe(e)

    # transformer encoder (BS, TL, E_DIM)
    transformer_out = self.transformer_encoder(e_embedded, src_key_padding_mask=padding_mask_encoder)

    # average over the trace length dim 1 (BS, E_DIM) ...
    # t = torch.mean(transformer_out, dim=1, keepdim=False)
    # ... or, use Conv1d
    # t = self.conv(transformer_out).squeeze()
    # ... or, take the last item
    # t = transformer_out[:,-1,:]
    # ... or, use AttentionPooling
    t = self.attn_pool(transformer_out)
    # ... or, use LSTM
    # t = self.lstm(transformer_out)
    # t = t[0][:,-1]

    # concat conditional label (BS, E_DIM+C_DIM)
    t = torch.cat((t, c), dim=1)

    # map t to mean and var vectors (BS, Z_DIM)
    return self.linear_mean(t), self.linear_var(t)


class Decoder(nn.Module):
  def __init__(self, trace_attributes, num_activities, num_resources, max_trace_length,
               d_dim, act_e_dim, res_e_dim, z_dim, c_dim,
               transformer_nhead, transformer_dim_feedforward, transformer_num_layers, dropout_p,
               tot_attr_e_dim,
               encoder_act2e, encoder_res2e, autoregressive_training, teacher_forcing_word_dropout_p,
               device):
    super(Decoder, self).__init__()

    self.trace_attributes = trace_attributes
    self.num_activities = num_activities
    self.num_resources = num_resources
    self.max_trace_length = max_trace_length
    self.has_trace_attributes = tot_attr_e_dim > 0
    self.d_dim = d_dim
    # self.d_dim_cat = d_dim + z_dim + c_dim # required if injecting latent space by concatenation
    self.encoder_act2e = encoder_act2e
    self.autoregressive_training = autoregressive_training
    self.teacher_forcing_word_dropout_p = teacher_forcing_word_dropout_p
    self.device = device

    self.padding_idx = num_activities

    self.dropout = nn.Dropout(p=dropout_p)
    self.pe = PositionalEncoding(d_dim)

    self.z2z_up = nn.Linear(z_dim+c_dim, d_dim)

    self.transformer_decoder = nn.TransformerDecoder(
      decoder_layer=nn.TransformerDecoderLayer(
        d_model=self.d_dim,
        nhead=transformer_nhead,
        dim_feedforward=transformer_dim_feedforward,
        dropout=dropout_p,
        batch_first=True,
      ),
      num_layers=transformer_num_layers,
    )

    self.e2a = nn.Linear(self.d_dim, num_activities)

  def apply_word_dropout(self, x, p=0.5):
    mask = torch.rand(x.shape).to(self.device) < p
    dropped = x.clone()
    dropped = dropped.to(self.device)
    dropped[mask] = self.padding_idx
    return dropped

  # if x is None, then we are at inference time (decoder is autoregressive)
  # if x is not None, then we are at training time (decoder uses teacher forcing, not autoregressive)
  def forward(self, z, c, x=None):
    if x:
      attrs, acts, ts, ress = x
    
    # concat conditional label
    z = torch.cat((z, c), dim=1)

    # upsample z
    z_up = self.dropout(F.relu(self.z2z_up(z)))

    # prepare memory for transformer decoder
    # memory = z_up.unsqueeze(1).repeat(1, self.max_trace_length, 1)
    # why the following works?
    # memory = z_up.unsqueeze(1)
    # use zero memory: if latent space is injected in some other way
    memory = torch.zeros(z.shape[0], self.max_trace_length, self.d_dim).to(self.device)

    if x and not self.autoregressive_training: # training (teacher forcing)
      # prepare target sequence
      acts_shifted = torch.roll(acts, 1, dims=1)                    # shift right
      acts_shifted[:,0] = self.num_activities - 1                   # add EoT activity at the beginning
      tgt_key_padding_mask = (acts_shifted == self.padding_idx).to(self.device)     # prepare padding mask
      acts_shifted = self.apply_word_dropout(acts_shifted, p=self.teacher_forcing_word_dropout_p)   # apply word dropout
      acts_shifted = self.encoder_act2e(acts_shifted)               # embed activities
      acts_shifted = self.pe(acts_shifted)                          # apply PE

      acts_shifted += z_up.unsqueeze(1).repeat(1, self.max_trace_length, 1)                                          # inject latent space by addition
      # acts_shifted = torch.cat((acts_shifted, z.unsqueeze(1).expand(-1, self.max_trace_length, -1)), dim=2)            # inject latent space by concatenation

      # prepare mask for target sequence
      tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_trace_length) # causal mask to disallow tokens to attend to future tokens
      tgt_mask = tgt_mask.to(self.device)

      # transformer decoder
      transformer_out = self.transformer_decoder(
        acts_shifted,
        memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask
      )

      # reconstruct activities
      acts_rec = F.softmax(self.e2a(transformer_out), dim=2)

      return acts_rec
    else: # inference (autoregressive)
      sot_input = self.encoder_act2e(torch.tensor([self.num_activities-1], dtype=torch.int64).to(self.device)).repeat(z.shape[0], 1)

      e_input = sot_input.unsqueeze(1)

      decoder_act_outputs = []

      for _ in range(self.max_trace_length):
        # positional encoding
        e_input_pe = self.pe(e_input)

        e_input_pe += z_up.unsqueeze(1)          # inject latent space by addition
        # e_input_pe = torch.cat((e_input_pe, z.unsqueeze(1).expand(-1, e_input_pe.shape[1], -1)), dim=2) # inject latent space by concat

        # transformer
        transformer_out = self.transformer_decoder(e_input_pe, memory)

        # reconstruct act
        act_rec = self.e2a(transformer_out[:,-1,:])
        act_rec = act_rec[:, None, :]
        decoder_act_outputs.append(act_rec)
        act_rec = act_rec.argmax(dim=2)
        act_rec = self.encoder_act2e(act_rec)

        # concat into event e
        e_rec = torch.cat([act_rec], dim=2)

        # concat e_rec into e_input
        e_input = torch.cat([e_input, e_rec], dim=1)

      acts_rec = torch.cat(decoder_act_outputs, dim=1)
      acts_rec = F.softmax(acts_rec, dim=2)

      return acts_rec


class VAETransformer(nn.Module):
  def __init__(self, trace_attributes=[], num_activities=12, num_resources=5, max_trace_length=10,
               transformer_nhead=1, transformer_dim_feedforward=2048, transformer_num_layers=1,
               attr_e_dim=4, act_e_dim=3, res_e_dim=3, z_dim=20, c_dim=0,
               dropout_p=0.1, autoregressive_training=False, teacher_forcing_word_dropout_p=0.5, device='cpu'):
    
    super(VAETransformer, self).__init__()

    self.z_dim = z_dim

    d_dim = sum([act_e_dim])
    self.d_dim = d_dim

    # encoder
    self.encoder = Encoder(
      trace_attributes=trace_attributes,
      num_activities=num_activities,
      num_resources=num_resources,
      max_trace_length=max_trace_length,
      transformer_nhead=transformer_nhead,
      transformer_dim_feedforward=transformer_dim_feedforward,
      transformer_num_layers=transformer_num_layers,
      d_dim=d_dim,
      attr_e_dim=attr_e_dim,
      act_e_dim=act_e_dim,
      res_e_dim=res_e_dim,
      z_dim=z_dim,
      c_dim=c_dim,
      dropout_p=dropout_p,
    )

    # decoder
    self.decoder = Decoder(
      trace_attributes=trace_attributes,
      num_activities=num_activities,
      num_resources=num_resources,
      max_trace_length=max_trace_length,
      d_dim=d_dim,
      act_e_dim=act_e_dim,
      res_e_dim=res_e_dim,
      z_dim=z_dim,
      c_dim=c_dim,
      tot_attr_e_dim=0,
      transformer_nhead=transformer_nhead,
      transformer_dim_feedforward=transformer_dim_feedforward,
      transformer_num_layers=transformer_num_layers,
      dropout_p=dropout_p,
      encoder_act2e=self.encoder.embed_acts,
      encoder_res2e=None,
      autoregressive_training=autoregressive_training,
      teacher_forcing_word_dropout_p=teacher_forcing_word_dropout_p,
      device=device
    )

    self.times_enc = []
    self.times_dec = []


  # p(z|x)
  def encode(self, x, c=None):
    return self.encoder(x, c)

  
  # p(x|z)
  def decode(self, z, c, x=None):
    return self.decoder(z, c, x)

  
  def forward(self, x, c=None):
    start_enc = time.time()
    mean, var = self.encode(x, c)
    end_enc = time.time()

    epsilon = torch.randn_like(var)
    z = mean + var*epsilon # reparametrization trick
    
    start_dec = time.time()
    out = self.decode(z, c, x)
    end_dec = time.time()

    self.times_enc.append(end_enc-start_enc)
    self.times_dec.append(end_dec-start_dec)

    # if len(self.times_enc) % 10 == 0:
    #   print(f'Encoder: {sum(self.times_enc) / len(self.times_enc)}s')
    #   print(f'Decoder: {sum(self.times_dec) / len(self.times_dec)}s')
    
    return out, mean, var