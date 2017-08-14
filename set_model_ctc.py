import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.utils.rnn as rnn_util
import torch.nn.functional as F
from collections import OrderedDict



class Seq(nn.Sequential):
  def __init__(self, *objs):
     super(Seq, self).__init__(*objs)

  def forward(self, input, batch_seq_len):
      for module in self._modules.values():
           input = module(input, batch_seq_len)
      return input




class Var_to_packed(nn.Module):
   def __init__(self):
       """
         convert a variable to a packed padded sequence
       """
       super(Var_to_packed, self).__init__()

    
   def forward(self, x, batch_sizes_t, batch_seq_len):
       """
         x: input variable
         batch_sizes_t: batch size at each time instant
         batch_seq_len: sequence length in a batch
       """
       start = 0
       batch_size_t_max = batch_sizes_t[0]
       out = []     
       for batch_len in batch_sizes_t:
           x_t = x[start : start+batch_len].contiguous()
           num_pad = batch_size_t_max - batch_len
           
           if num_pad > 0:
              x_t = x_t.view(1,1,x_t.size(0), x_t.size(1))
              x_t = F.pad(x_t, pad = (0, 0, 0, num_pad), mode = 'constant', value = 0).squeeze()
           out.append(x_t)
           start += batch_len
       out = torch.stack(out, dim = 0)     # (t, n, h)
       out = out.transpose(0,1).contiguous()                # (n, t, h)
       out = rnn_util.pack_padded_sequence(out, batch_seq_len, batch_first=True)
       return out




class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        convert a packed padded_sequence to a variable, goes through the forward process, and convert back to a packed padded sequence
        """
        super(SequenceWise, self).__init__()
        self.module = module
        self.var_to_packed = Var_to_packed()

    def forward(self, x, batch_seq_len):
        """
        x is a packed padded sequence
        """
        x, batch_sizes_t = x.data, x.batch_sizes
        x = self.module(x)       # pass the module processing
        x = self.var_to_packed(x, batch_sizes_t, batch_seq_len)     # convert to packed padded sequence
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr




class InferenceBatchLogSoftmax(nn.Module):

    def forward(self, x, batch_seq_before):     # x is a packed padded variable
        x, batch_seq_after = rnn_util.pad_packed_sequence(x, batch_first = False)      # x size (T, N, H)
        T_max = x.size(0)        

        if batch_seq_after != batch_seq_before:
            raise Exception('batch sequence length is wrong')

        if not self.training:
            return torch.stack([F.log_softmax(x[i]) for i in range(T_max)], 0)
        else:
            return x






class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False, batch_first = True)

    def forward(self, x, batch_seq_len):
        # x is a packed padded sequence
        if self.batch_norm is not None:
            x = self.batch_norm(x, batch_seq_len)
        x, _ = self.rnn(x)
        return x






class Layered_RNN(nn.Module):
    def __init__(self, rnn_input_size = 40, rnn_type=nn.LSTM, rnn_hidden_size=768, nb_layers=5, bidirectional=True, batch_norm = True, num_classes = 1000):
        super(Layered_RNN, self).__init__()
        num_directions = 2 if bidirectional else 1


        rnns = []      # hold each layer of the RNN
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))

        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size = num_directions * rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm = batch_norm)     # input to intermediate layers are batch normed
            rnns.append(('%d' % (x + 1), rnn))

        self.rnns = Seq(OrderedDict(rnns))
        
        if batch_norm:
           fully_connected = nn.Sequential(
               nn.BatchNorm1d(num_directions * rnn_hidden_size),
               nn.Linear(num_directions * rnn_hidden_size, num_classes + 1, bias=False)
           )
        else:
           fully_connected = nn.Linear(num_directions * rnn_hidden_size, num_classes + 1, bias=False)

        self.fc = SequenceWise(fully_connected)
        self.inference_log_softmax = InferenceBatchLogSoftmax()


    def forward(self, x, batch_seq_len):
        x = self.rnns(x, batch_seq_len)
        x = self.fc(x, batch_seq_len)
        
        # identity in training mode, logsoftmax in eval mode
        x = self.inference_log_softmax(x, batch_seq_len)
        
        return x



