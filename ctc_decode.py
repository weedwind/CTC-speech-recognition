import torch


class Decoder(object):

   def __init__(self, space_idx = -1):
      self.space_idx = space_idx        # space_idx = -1 means no space symbol


   def greedy_decoder(self, prob_tensor, frame_seq_len, target_labels, label_len):
      """
        prob_tensor: a tensor contains probabilities, size (t, n, c)
        frame_seq_len: a list contains frame level sequence length
        target_labels: a list contains target transcriptions
        label_len: a list contains target label length
      """
      prob_tensor = prob_tensor.transpose(0,1)         # (n, t, c)
      _, decoded = torch.max(prob_tensor, 2)
      decoded = decoded.view(decoded.size(0), decoded.size(1))
      decoded = self._process_seqs(decoded, frame_seq_len, remove_rep = True)
      target_labels = self._unflatten_targets(target_labels, label_len)
      return self._distances(decoded, target_labels)


   def _unflatten_targets(self, targets, target_sizes):
       split_targets = []
       offset = 0

       for size in target_sizes:
          split_targets.append(targets[offset : offset + size])
          offset += size
       return split_targets


   def _process_seqs(self, seqs, seq_lens, remove_rep = False):
      results = []
      if seqs.size(0) != len(seq_lens):
         raise Exception('number of seqs in this batch is wrong')
      for i in range(seqs.size(0)):
         results.append(self._process_seq_i(seqs[i,:], seq_lens[i], remove_rep))
      return results

   
   def _process_seq_i(self, seq, seq_len, remove_rep = False):
       result = []

       for i, char in enumerate(seq[0 : seq_len]):
          if char != 0:       # 0 is the blank index
             if remove_rep and i != 0 and char == seq[i - 1]:   # duplicates
                pass
             else:
                result.append(char)

       if not result: return result

       if self.space_idx > 0:
          while True:
            if not result: break

            if result[0] == self.space_idx:
               result.pop(0)
            elif result[-1] == self.space_idx:
               result.pop()
            else:
               break
               
       return result


   def _convert_to_chars(self, digit_seqs, labels):
      int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])     # mapping from numerical index to character labels
      char_seqs = []

      for i in range(len(digit_seqs)):
         char_seqs.append(self._convert_to_char(digit_seqs[i], int_to_char))
      
      return char_seqs



   def _convert_to_char(self, digit_seq_i, int_to_char):
      if digit_seq_i:
         return [int_to_char[d] for d in digit_seq_i]
      else:
         return []

   
   def _convert_to_words(self, char_seqs):
      word_seqs = []
      for i in range(len(char_seqs)):
         word_seqs.append(self._convert_to_word(char_seqs[i]))
      return word_seqs



   def _convert_to_word(self, char_seq_i):         # convert a list of characters to a list of words
      if char_seq_i:
         return ''.join(char_seq_i).strip().split()
      else:
         return []
 

   def _distances(self, src_seqs, tgt_seqs):
      if len(src_seqs) != len(tgt_seqs):
         raise Exception('number of source and target sequences are not equal')
      dist = 0
      for i in range(len(src_seqs)):
          dist += self._distance_i(src_seqs[i], tgt_seqs[i])
      return dist



   def _distance_i(self, src_seq, tgt_seq):      # compute edit distance between two iterable objects
      L1, L2 = len(src_seq), len(tgt_seq)
      if L2 == 0:
         raise Exception('target sequence is empty')

      if L1 == 0: return L2

      # construct matrix of size (L1 + 1, L2 + 1)

      dist = [[0] * (L2 + 1) for i in range(L1 + 1)]

      for i in range(1, L2 + 1):
         dist[0][i] = dist[0][i-1] + 1

      for i in range(1, L1 + 1):
         dist[i][0] = dist[i-1][0] + 1

      for i in range(1, L1 + 1):
         for j in range(1, L2 + 1):
            if src_seq[i - 1] == tgt_seq[j - 1]:
                cost = 0
            else:
                cost = 1
            dist[i][j] = min(dist[i][j-1] + 1, dist[i-1][j] + 1, dist[i-1][j-1] + cost)

      return dist[L1][L2]




class GreedyDecoder_test(Decoder):
   def __init__(self, labels, output = 'char', space_idx = -1):
      """
        labels: a list contains all the CTC labels in the same order as the CTC outputs, including blank label
        output: 'char' or 'word'
        space_idx: the position of the space, -1 means no space
      """
      super(GreedyDecoder_test, self).__init__(space_idx)
      self.labels = labels
      self.output = output


   def decode(self, prob_tensor, frame_seq_len):
      """
        prob_tensor: a tensor contains probabilities, size (t, n, c)
        frame_seq_len: a list contains frame level sequence length
      """
      prob_tensor = prob_tensor.transpose(0,1)         # (n, t, c)
      _, decoded = torch.max(prob_tensor, 2)
      decoded = decoded.view(decoded.size(0), decoded.size(1))
      decoded = self._process_seqs(decoded, frame_seq_len, remove_rep = True)
      decoded = self._convert_to_chars(decoded, self.labels)     # convert digit idx to chars
      if self.output == 'word':
         decoded = self._convert_to_words(decoded)
      return decoded





class BeamDecoder_test(Decoder):
  def __init__(self, labels, scorer, top_paths = 1, beam_width = 20, output = 'char', space_idx = -1):
     super(BeamDecoder_test, self).__init__(space_idx)
     self.labels = labels
     self.output = output
     assert top_paths == 1, "Only supports top 1 path in the current version"

     try:
        import pytorch_ctc
     except ImportError:
        raise ImportError("BeamCTCDecoder requires pytorch_ctc package.")

     self._decoder = pytorch_ctc.CTCBeamDecoder(scorer = scorer, labels = self.labels, top_paths = top_paths, beam_width = beam_width, blank_index = 0, space_index = self.space_idx, merge_repeated=False)

  def decode(self, prob_tensor, frame_seq_len):
    """
      prob_tensor: a tensor contains log probabilities, size (t, n, c)
      frame_seq_len: a list contains frame level sequence length
    """
    frame_seq_len = torch.IntTensor(frame_seq_len).cpu()
    decoded, _, out_seq_len = self._decoder.decode(prob_tensor, seq_len = frame_seq_len)
    decoded = decoded[0]
    out_seq_len = out_seq_len[0]
    decoded = self._process_seqs(decoded, out_seq_len, remove_rep = False)
    decoded = self._convert_to_chars(decoded, self.labels)     # convert digit idx to chars
    if self.output == 'word':
       decoded = self._convert_to_words(decoded)
    return decoded

