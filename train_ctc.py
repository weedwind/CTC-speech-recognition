import torch
import numpy as np
import os
import glob
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import shutil
import collections
import set_model_ctc
from warpctc_pytorch import CTCLoss
from random import sample
import h5py
import gc
from ctc_decode import Decoder


gpu_dtype = torch.cuda.FloatTensor


train_dir_base = 'train_ctc'
val_dir_base = 'cv_ctc'

train_files = glob.glob(train_dir_base + '/' + '*.h5')
val_files = glob.glob(val_dir_base + '/' + '*.h5')
train_num_chunks = len(train_files)
val_num_chunks = len(val_files)

load_chunks_every = 10     # everytime load this many of .h5 data chunks into memory

layers = 4          # number of LSTM layers
hidden_size = 256   # number of cells per direction
num_dirs = 2        # number of LSTM directions



assert num_dirs == 1 or num_dirs == 2, 'num_dirs must be 1 or 2'


def _collate_fn(batch):

   def func(p):
      """
        p is a tuple, p[0] is the data tensor, p[1] is the target seq label list
        data_tensor: (T, F)
      """
      return p[0].size(0)

   batch = sorted(batch, reverse = True, key = func)
   longest_sample = batch[0][0]
   feat_size = longest_sample.size(1)
   minibatch_size = len(batch)
   max_seqlength = longest_sample.size(0)
   inputs = torch.zeros(minibatch_size, max_seqlength, feat_size)
   input_sizes =  torch.IntTensor(minibatch_size)
   target_sizes = torch.IntTensor(minibatch_size)
   targets = []
   input_sizes_list = []
   for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(tensor)
        input_sizes[x] = seq_length
        input_sizes_list.append(seq_length)
        target_sizes[x] = len(target)
        targets.extend(target)
   targets = torch.IntTensor(targets)
   return inputs, targets, input_sizes, input_sizes_list, target_sizes





def get_loader(chunk_list, mode):
  class MyDataset(torch.utils.data.Dataset):
      def __init__(self):
          self.data_files = {}
          i = 0
          for f in chunk_list:
             if mode == 'train': print ('Loading data from %s' %f)
             with h5py.File(f, 'r') as hf:
                for grp in hf:
                    self.data_files[i] = (torch.FloatTensor(np.asarray(hf[grp]['data'])), list(map(int, list(np.asarray(hf[grp]['label'])))))
                    i += 1
          if mode == 'train': print ('Total %d sequences loaded' %len(self.data_files))

      def __getitem__(self, idx):
          return self.data_files[idx]

      def __len__(self):
          return len(self.data_files)

  dset = MyDataset()
  if mode == 'train':
      loader = DataLoader(dset, batch_size = 4, shuffle = True, collate_fn = _collate_fn, num_workers = 10, pin_memory = False)
  elif mode == 'test':
      loader = DataLoader(dset, batch_size = 4, shuffle = False, collate_fn = _collate_fn, num_workers = 10, pin_memory = False)
  else:
      raise Exception('mode can only be train or test')

  return loader





def train_one_epoch(model, loss_fn, optimizer, print_every = 10):
    data_list = sample(train_files, train_num_chunks)
    model.train()
    t = 0
    
    for i in range(0, train_num_chunks, load_chunks_every):
          chunk_list = data_list[i: i + load_chunks_every]
          loader_train = get_loader(chunk_list, 'train')
          for data in loader_train:
             inputs, targets, input_sizes, input_sizes_list, target_sizes = data
             batch_size = inputs.size(0)
             inputs = Variable(inputs, requires_grad=False).type(gpu_dtype)
             target_sizes = Variable(target_sizes, requires_grad=False)
             targets = Variable(targets, requires_grad=False)
             input_sizes = Variable(input_sizes, requires_grad = False)
             
             inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list, batch_first=True)
             
       
             out = model(inputs, input_sizes_list)
     
             loss = loss_fn(out, targets, input_sizes, target_sizes)     # ctc loss
             loss /= batch_size

             if (t + 1) % print_every == 0:
                 print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
             #if (t + 1 ) % 50 == 0:  check_accuracy(model)
             optimizer.zero_grad()
             loss.backward()

             torch.nn.utils.clip_grad_norm(model.parameters(), 400)     # clip gradients
             optimizer.step()
             t += 1
          loader_train.dataset.data_files.clear()
          del loader_train
          gc.collect()       
 




def check_accuracy(model):
    total_num_errs = 0
    total_num_tokens = 0
    decoder = Decoder(space_idx = -1)
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    data_list = val_files

    for i in range(0, val_num_chunks, load_chunks_every):
       chunk_list = data_list[i: i + load_chunks_every]
       loader_val = get_loader(chunk_list, 'test')
       for data in loader_val:
           inputs, targets, input_sizes, input_sizes_list, target_sizes = data
           inputs = Variable(inputs, volatile = True, requires_grad=False).type(gpu_dtype)
           inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list, batch_first=True)
           probs = model(inputs, input_sizes_list)
           probs = probs.data.cpu()
           total_num_errs +=  decoder.greedy_decoder(probs, input_sizes_list, targets, target_sizes)
           total_num_tokens += sum(target_sizes)
       loader_val.dataset.data_files.clear()
       del loader_val
       gc.collect()
    acc = 1 - float(total_num_errs) / total_num_tokens
    print('Phone accuracy = %.2f' %(100 * acc ,))
    return 100 * acc




def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay




def train_epochs(model, loss_fn, init_lr, model_dir):
   if os.path.exists(model_dir):
      shutil.rmtree(model_dir)
   os.makedirs(model_dir)

   optimizer = optim.Adam(model.parameters(), lr = init_lr)     # setup the optimizer

   learning_rate = init_lr

   max_iter = 50                 # maximum number of epochs
   end_halfing_inc = 0.05        # if cv accuracy increases by less than this threshold, stop training
   start_halfing_inc = 0.25      # if cv accuracy increases by less than this threshold, begin decreasing learning rate
   halfing_factor = 0.1          # new learning rate = current learning rate * halfing_factor

   acc_best = -99999
   count = 0                     # epoch counter
   best_model_state = None
   best_op_state = None
   half_flag = False
   stop_train = False

   while not stop_train:
     if count > max_iter: break
     count += 1
     print ("Starting epoch", count)

     if half_flag:
        learning_rate *= halfing_factor
        adjust_learning_rate(optimizer, halfing_factor)     # decay learning rate

     train_one_epoch(model, loss_fn, optimizer)      # train one epoch
     acc = check_accuracy(model)        # check accuracy
     model_path_accept = model_dir + '/epoch' + str(count) + '_lr' + str(learning_rate) + '_cv' + str(acc) + '.pkl'
     model_path_reject = model_dir + '/epoch' + str(count) + '_lr' + str(learning_rate) + '_cv' + str(acc) + '_rejected.pkl'

     if acc > (acc_best + start_halfing_inc):    # accept model
        #if half_flag: half_flag = False
        best_model_state = model.state_dict()
        best_op_state = optimizer.state_dict()
        acc_best = acc
        model_path = model_path_accept
        torch.save(model.state_dict(), model_path)
     elif (acc > acc_best) and (not half_flag):                                   # accept model but decay learning rate
        half_flag = True
        best_model_state = model.state_dict()
        best_op_state = optimizer.state_dict()
        acc_best = acc
        model_path = model_path_accept
        torch.save(model.state_dict(), model_path)
     elif (acc <= acc_best) and (not half_flag):       # do not accept the model and decay learning rate
        model_path = model_path_reject
        torch.save(model.state_dict(), model_path)
        half_flag = True
        model.load_state_dict(best_model_state)             # model back up to the previous best one
        optimizer.load_state_dict(best_op_state)               # optimizer back up to the previous best one
     elif half_flag:
        if acc > (acc_best + end_halfing_inc):       # still accept model
            best_model_state = model.state_dict()
            best_op_state = optimizer.state_dict()
            acc_best = acc
            model_path = model_path_accept
            torch.save(model.state_dict(), model_path)
        else:
            if acc > acc_best:
               best_model_state = model.state_dict()
               best_op_state = optimizer.state_dict()
               acc_best = acc
               model_path = model_path_accept
               torch.save(model.state_dict(), model_path)
               stop_train = True
            else:
               model_path = model_path_reject
               torch.save(model.state_dict(), model_path)
               model.load_state_dict(best_model_state)
               optimizer.load_state_dict(best_op_state)
               stop_train = True
   
   print ("End training, best cv accuracy is:", acc_best)
   best_path = model_dir + '/best_model' + '_cv' + str(acc_best) + '.pkl'
   torch.save(best_model_state, best_path)




if __name__ == '__main__':
   model = set_model_ctc.Layered_RNN(rnn_input_size = 40, nb_layers = layers, rnn_hidden_size = hidden_size, bidirectional = True if num_dirs==2 else False, batch_norm = True, num_classes = 61) #the model will add 1 to the num_classes for the blank label automatically
   model = model.type(gpu_dtype)
   loss_fn = CTCLoss()
   train_epochs(model, loss_fn,  1e-3, 'weights_ctc')



