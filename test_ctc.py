import numpy as np
import torch
import set_model_ctc
import htk_io
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import ctc_decode

gpu_dtype = torch.cuda.FloatTensor

eps = 1e-8


stat_file = "stat"      # mean and variance of the training data will be extracted from this file
mapping_file = 'hmmlist'  # a list of phones, in the same order as the CTC output, excluding the blank label
layers = 4
hidden_size = 256
num_dirs = 2
decoder_type = 'Beam'      # Beam or Beam_LM or Greedy
out_mlf = 'result.mlf'     # output mlf file, containing the decoded strings

if decoder_type == 'Beam' or decoder_type == 'Beam_LM':
   try:
     import pytorch_ctc
     from pytorch_ctc import Scorer, KenLMScorer
   except ImportError:
     print("warn: pytorch_ctc unavailable. Only greedy decoding is supported.")
elif decoder_type != 'Greedy':
   raise Exception('decoder_type must be Beam, Beam_LM or Greedy')


if decoder_type == 'Beam_LM':
  # Timit labels phones, not letter, we have to convert the phone to single-char symbols and put them in the labels string, following the same order as the CTC outputs, including the blank label
  # See map_list for the correspondence between TIMIT phone and these symbols
  labels = '_123456789abcde~-hij,.|{ofg?!+u}[x]@ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  dict_path = 'symbol_list'  # This is the vocabulary. In TIMIT, the word vocabulary is the same as the phones, also the same as the symbols (excluding the blank lable)
  kenlm_path = 'bigram.ken'   # This is the kenlm converted from the ARPA bigram using build_binary provided by kenlm.
  trie_path  = 'trie'         # output trie will be saved here
  lm_weight = 2.1
  lm_beta1 = 1
  lm_beta2 = 1
  pytorch_ctc.generate_lm_trie(dict_path, kenlm_path, trie_path, labels, 0, -1)




def create_mapping(map_file):
   labels = ['_']

   with open(map_file) as m:
      for line in m:
         line = line.strip()
         if line:
            labels.append(line)
   return labels



def read_mv(stat):
   mean_flag = var_flag = False
   m = v = None
   with open(stat) as s:
      for line in s:
         line = line.strip()
         if len(line) < 1: continue
         if "MEAN" in line:
            mean_flag = True
            continue
         if mean_flag:
            m = list(map(float, line.split()))
            mean_flag = False
            continue
         if "VARIANCE" in line:
            var_flag = True
            continue
         if var_flag:
            v = list(map(float, line.split()))
            var_flag = False
            continue
   return np.array(m, dtype = np.float64), np.array(v, dtype = np.float64)



def org_data(utt_feat, skip_frames = 0 ):
   num_frames, num_channels = utt_feat.shape

   if skip_frames > 0:
      utt_feat = np.pad(utt_feat, ((0, skip_frames), (0,0)), mode = 'edge')    # pad the ending frames
      utt_feat = utt_feat[skip_frames:,:]

   return utt_feat.reshape(1, num_frames, num_channels)



def gen_decoded(feat_list, model_path):
   model = set_model_ctc.Layered_RNN(rnn_input_size = 40, nb_layers = layers, rnn_hidden_size = hidden_size, bidirectional = True if num_dirs==2 else False, batch_norm = True, num_classes = 61)
   model = model.type(gpu_dtype)
   model.load_state_dict(torch.load(model_path))     # load model params
   model.eval()             # Put the model in test mode (the opposite of model.train(), essentially)
   
   if decoder_type == 'Greedy':
      labels = create_mapping(mapping_file)
      decoder = ctc_decode.GreedyDecoder_test(labels, output = 'char', space_idx = -1)     # setup greedy decoder
   if decoder_type == 'Beam':
      labels = create_mapping(mapping_file)
      scorer = Scorer()
      decoder = ctc_decode.BeamDecoder_test(labels, scorer, top_paths = 1, beam_width = 200, output = 'char', space_idx = -1)    # setup beam decoder without lm
   if decoder_type == 'Beam_LM':
      labels_symbol = '_123456789abcde~-hij,.|{ofg?!+u}[x]@ABCDEFGHIJKLMNOPQRSTUVWXYZ'
      labels_true = create_mapping(mapping_file)
      # need to use the fake symbols here for consistency with the trie
      scorer = KenLMScorer(labels_symbol, kenlm_path, trie_path, blank_index = 0, space_index = -1)
      scorer.set_lm_weight(lm_weight)
      scorer.set_word_weight(lm_beta1)
      scorer.set_valid_word_weight(lm_beta2)
      # need to use the true timit label to convert the decoded position indexes back to phone labels
      decoder = ctc_decode.BeamDecoder_test(labels_true, scorer, top_paths = 1, beam_width = 200, output = 'char', space_idx = -1)    # setup beam decoder with lm
   
   m, v = read_mv(stat_file)
   if m is None or v is None:
      raise Exception("mean or variance vector does not exist")

   with open(feat_list) as f:
      with open(out_mlf, 'w') as fw:
        fw.write('#!MLF!#\n')
        for line in f:
           line = line.strip()
           if len(line) < 1: continue
           print ("recognizing file %s" %line)
           out_name = '"' + line[:line.rfind('.')] + '.rec' + '"'
           fw.write(out_name + '\n')
           io = htk_io.fopen(line)
           utt_feat = io.getall()
           utt_feat -= m       # normalize mean
           utt_feat /= (np.sqrt(v) + eps)     # normalize var
           feat_numpy = org_data(utt_feat, skip_frames = 5)
           feat_tensor = torch.from_numpy(feat_numpy).type(gpu_dtype)
           x = Variable(feat_tensor.type(gpu_dtype), volatile = True)
           input_sizes_list = [x.size(1)]
           x = nn.utils.rnn.pack_padded_sequence(x, input_sizes_list, batch_first=True)         
           probs = model(x, input_sizes_list)
           probs = probs.data.cpu()
           decoded = decoder.decode(probs, input_sizes_list)[0]
           for word in decoded:
             fw.write(word + '\n')
           fw.write('.\n')
           print (' '.join(decoded))



if __name__ == '__main__':
    gen_decoded(feat_list = 'core_fea.scp', model_path = 'weights_ctc/best_model_cv75.0799539465268.pkl')
