import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
  def __init__(self, trace_attributes, num_activities, num_lstm_layers,
               attr_e_dim, act_e_dim, cf_dim, z_dim, c_dim, dropout_p,
               is_conditional):
    super(Encoder, self).__init__()

    self.trace_attributes = trace_attributes
    self.is_conditional = is_conditional

    self.dropout = nn.Dropout(p=dropout_p)

    self.attr2e = nn.ModuleDict()
    self.tot_attr_e_dim = 0
    for trace_attr in trace_attributes:
      if trace_attr['type'] == 'categorical':
        self.attr2e[trace_attr['name']] = nn.Embedding(
          num_embeddings=len(trace_attr['possible_values']),
          embedding_dim=attr_e_dim,
        )
        self.tot_attr_e_dim += attr_e_dim
      elif trace_attr['type'] == 'numerical':
        self.tot_attr_e_dim += 1
      else:
        raise Exception(f'Unknown trace attribute type: {trace_attr["type"]}')
      
    self.has_trace_attributes = self.tot_attr_e_dim > 0

    self.attrs2attrs = nn.Linear(self.tot_attr_e_dim, self.tot_attr_e_dim // 2)

    self.act2e = nn.Embedding(
      num_embeddings=num_activities+1,
      embedding_dim=act_e_dim,
      padding_idx=num_activities,
    )

    self.e2cf = nn.LSTM(
      input_size=act_e_dim+1,
      hidden_size=cf_dim,
      num_layers=num_lstm_layers,
      dropout=dropout_p if num_lstm_layers > 1 else 0,
      batch_first=True,
    )

    t_dim = self.tot_attr_e_dim//2 + cf_dim
    self.t_dim = t_dim

    self.t2mean = nn.Linear(t_dim+c_dim if is_conditional else t_dim, z_dim)
    self.t2var = nn.Linear(t_dim+c_dim if is_conditional else t_dim, z_dim)

  def forward(self, x, c=None):
    attrs, acts, ts = x

    # embed attrs
    e_attrs = []
    for trace_attr in self.trace_attributes:
      if trace_attr['type'] == 'categorical':
        e_attr = self.attr2e[trace_attr['name']](attrs[trace_attr['name']])
      elif trace_attr['type'] == 'numerical':
        e_attr = attrs[trace_attr['name']].unsqueeze(1)
        e_attr = e_attr.float()
      else:
        raise Exception(f'Unknown trace attribute type: {trace_attr["type"]}')
        
      e_attrs.append(e_attr)

    if len(e_attrs) == 0:
      e_attrs.append(torch.tensor([]))

    e_attrs = torch.cat((*e_attrs,), dim=1)
    e_attrs = self.dropout(F.relu(self.attrs2attrs(e_attrs)))

    # embed acts
    e_acts = self.act2e(acts)

    # concat ts
    ts = ts.unsqueeze(dim=2)
    e_acts = torch.cat((e_acts, ts), dim=2)

    # forward acts to lstm
    lens = acts.argmax(dim=1)
    e_packed = pack_padded_sequence(e_acts, lens.to('cpu'), batch_first=True, enforce_sorted=False)
    cf_packed, _ = self.e2cf(e_packed)
    cf, _ = pad_packed_sequence(cf_packed, batch_first=True)
    cf = cf[torch.arange(cf.shape[0]), lens-1] # for each batch, take only the last LSTM output
    cf = self.dropout(cf)

    # concat attrs and cf
    t = torch.cat((e_attrs, cf), dim=1)

    # concat conditional label
    if self.is_conditional:
      t = torch.cat((t, c), dim=1)

    return self.t2mean(t), self.t2var(t)


class Decoder(nn.Module):
  def __init__(self, trace_attributes, num_activities, max_trace_length, num_lstm_layers,
               act_e_dim, cf_dim, t_dim, z_dim, c_dim, dropout_p,
               tot_attr_e_dim, is_conditional,
               encoder_act2e, device):
    super(Decoder, self).__init__()

    self.trace_attributes = trace_attributes
    self.num_activities = num_activities
    self.max_trace_length = max_trace_length
    self.num_lstm_layers = num_lstm_layers
    self.is_conditional = is_conditional
    self.has_trace_attributes = tot_attr_e_dim > 0
    self.cf_dim = cf_dim
    self.t_dim = t_dim
    self.encoder_act2e = encoder_act2e
    self.device = device

    self.dropout = nn.Dropout(p=dropout_p)

    self.z2t = nn.Linear(z_dim+c_dim if is_conditional else z_dim, t_dim)

    self.t2attr = nn.ModuleDict()
    for trace_attr in self.trace_attributes:
      if trace_attr['type'] == 'categorical':
        self.t2attr[trace_attr['name']] = nn.Sequential(
          nn.Linear(t_dim, t_dim//2),
          nn.ReLU(),
          self.dropout,
          nn.Linear(t_dim//2, len(trace_attr['possible_values'])),
        )
      elif trace_attr['type'] == 'numerical':
        self.t2attr[trace_attr['name']] = nn.Sequential(
          nn.Linear(t_dim, t_dim//2),
          nn.ReLU(),
          self.dropout,
          nn.Linear(t_dim//2, 1),
        )
      else:
        raise Exception(f'Unknown trace attribute type: {trace_attr["type"]}')
      
    self.t2e_act = nn.LSTM(
      input_size=t_dim+act_e_dim,
      hidden_size=cf_dim,
      num_layers=num_lstm_layers,
      dropout=dropout_p if num_lstm_layers > 1 else 0,
      batch_first=True,
    )

    self.t2e_ts = nn.LSTM(
      input_size=t_dim+act_e_dim+1,
      hidden_size=cf_dim,
      num_layers=num_lstm_layers,
      dropout=dropout_p if num_lstm_layers > 1 else 0,
      batch_first=True,
    )
    
    self.e2act = nn.Linear(cf_dim, num_activities)
    self.e2ts = nn.Sequential(
      nn.Flatten(start_dim=1),
      nn.Linear(cf_dim, cf_dim // 2),
      nn.ReLU(),
      self.dropout,
      nn.Linear(cf_dim // 2, 1),
    )


  def forward(self, z, c=None):
    # concat conditional label
    if self.is_conditional:
      z = torch.cat((z, c), dim=1)

    t_rec = self.dropout(F.relu(self.z2t(z)))

    # attrs
    attrs_rec = {}
    if self.has_trace_attributes:
      for trace_attr in self.trace_attributes:
        attr = self.t2attr[trace_attr['name']](t_rec)
        attr = F.softmax(attr, dim=1) if trace_attr['type'] == 'categorical' else F.sigmoid(attr)
        attrs_rec[trace_attr['name']] = attr

    # acts
    eot_input = self.encoder_act2e(torch.tensor([self.num_activities-1], dtype=torch.int64).to(self.device)).repeat(t_rec.shape[0], 1)
    ts_initial = torch.tensor([0.0]).repeat(t_rec.shape[0], 1).to(self.device)

    acts_lstm_input = torch.cat((t_rec, eot_input), dim=1).view(t_rec.shape[0], 1, -1)
    acts_lstm_hidden = (torch.zeros((self.num_lstm_layers, t_rec.shape[0], self.cf_dim)).to(self.device), torch.zeros((self.num_lstm_layers, t_rec.shape[0], self.cf_dim)).to(self.device))
    ts_lstm_input = torch.cat((t_rec, eot_input, ts_initial), dim=1).view(t_rec.shape[0], 1, -1)
    ts_lstm_hidden = (torch.zeros((self.num_lstm_layers, t_rec.shape[0], self.cf_dim)).to(self.device), torch.zeros((self.num_lstm_layers, t_rec.shape[0], self.cf_dim)).to(self.device))
    decoder_act_outputs, decoder_ts_outputs = [], []

    act_rec, ts_rec = eot_input, ts_initial
    for _ in range(self.max_trace_length):
      # acts lstm
      acts_lstm_output, acts_lstm_hidden = self.t2e_act(acts_lstm_input, acts_lstm_hidden)

      # acts "de-embedding"
      act_rec = self.e2act(acts_lstm_output)
      decoder_act_outputs.append(act_rec)
      act_rec = act_rec.view(act_rec.shape[0], -1).argmax(dim=1)
      act_rec = self.encoder_act2e(act_rec)

      # ts lstm
      ts_lstm_input = torch.cat((t_rec, act_rec, ts_rec), dim=1).view(t_rec.shape[0], 1, -1)
      ts_lstm_output, ts_lstm_hidden = self.t2e_ts(ts_lstm_input, ts_lstm_hidden)

      # ts reconstruction
      ts_rec = self.e2ts(ts_lstm_output)
      
      decoder_ts_outputs.append(ts_rec)

      # feed output of step t to input of step t+1
      acts_lstm_input = torch.cat((t_rec, act_rec), dim=1).view(t_rec.shape[0], 1, -1)
    
    acts_rec = torch.cat(decoder_act_outputs, dim=1)
    acts_rec = F.softmax(acts_rec, dim=2)

    ts_rec = torch.cat(decoder_ts_outputs, dim=1)

    return attrs_rec, acts_rec, ts_rec


class VAE(nn.Module):
  """
  PyTorch model for a VAE (c_dim = 0) or Conditional VAE (c_dim > 0)

  (attrs, acts) -> (e_attrs, e_acts) -> (e_attrs, cf) -> t -> z -> t_rec -> (e_attrs_rec, cf_rec) -> (e_attrs_rec, e_acts_rec) -> (attrs_rec, acts_rec)
  """
  def __init__(self, trace_attributes=[], num_activities=12, max_trace_length=10,
               num_lstm_layers=1, attr_e_dim=4, act_e_dim=3, cf_dim=5, z_dim=20, c_dim=0,
               dropout_p=0.1, device='cpu'):
    super(VAE, self).__init__()

    self.z_dim = z_dim
    self.is_conditional = False if c_dim == 0 else True

    # encoder
    self.encoder = Encoder(
      trace_attributes=trace_attributes,
      num_activities=num_activities,
      num_lstm_layers=num_lstm_layers,
      attr_e_dim=attr_e_dim,
      act_e_dim=act_e_dim,
      cf_dim=cf_dim,
      z_dim=z_dim,
      c_dim=c_dim,
      dropout_p=dropout_p,
      is_conditional=self.is_conditional,
    )

    # decoder
    self.decoder = Decoder(
      trace_attributes=trace_attributes,
      max_trace_length=max_trace_length,
      num_activities=num_activities,
      num_lstm_layers=num_lstm_layers,
      act_e_dim=act_e_dim,
      cf_dim=cf_dim,
      t_dim=self.encoder.t_dim,
      z_dim=z_dim,
      c_dim=c_dim,
      dropout_p=dropout_p,
      tot_attr_e_dim=self.encoder.tot_attr_e_dim,
      is_conditional=self.is_conditional,
      encoder_act2e=self.encoder.act2e,
      device=device,
    )

  # p(z|x)
  def encode(self, x, c=None):
    return self.encoder(x, c)

  # p(x|z)
  def decode(self, z, c=None):
    return self.decoder(z, c)

  def forward(self, x, c=None):
    mean, var = self.encode(x, c)
    epsilon = torch.randn_like(var)
    z = mean + var*epsilon # reparametrization trick
    
    return self.decode(z, c), mean, var
