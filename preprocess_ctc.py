import numpy as np
import htk_io
import os
import h5py
import gc

eps = 1e-8
stat_file = "stat"
mlf_file ="ref61.mlf"
mapping_file = 'hmmlist'

chunk_seq = 200
buffer_seq = 1000


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





def save_hd5(filename, obj):
   with h5py.File(filename, 'w') as hf:
      for i in range(len(obj)):
         grp = hf.create_group(str(i))
         grp.create_dataset('data', data = obj[i][0])
         grp.create_dataset('label', data = obj[i][1])
   print ('Saved %d sequences into %s' %(i+1, filename))




def make_chunk(filename, arr, arr_len):

   remain_arr = []
   dump_arr = []
   prob = float(chunk_seq) / arr_len

   for e in arr:
      flag_choose = np.random.binomial(1, prob)
      if flag_choose == 1:
         dump_arr.append(e)
         arr_len -= 1
      else:
         remain_arr.append(e)

   save_hd5(filename, dump_arr)

   del dump_arr[:]
   del dump_arr
   gc.collect()

   return remain_arr, arr_len





def create_mlf_dict(mlf):
   state2int = create_mapping(mapping_file)
   mlf_dict = {}
   with open(mlf, 'r') as f:
      for line in f:
         line = line.strip()
         if len(line) < 1: continue
         if line[0] == '"':     # a new utterance
              feat_filename = line.strip('"*/')[:-4] + '.fea'
              mlf_dict[feat_filename] = []
         elif line[0].isdigit():
            start, end, state = line.split()[:3]     # start, end frame index
            label = state2int[state]
            mlf_dict[feat_filename].append(label)
         elif line[0].isalpha():                     # no time label
            label = state2int[line]
            mlf_dict[feat_filename].append(label)
         elif line[0] == '.':              # end of a utterance
                mlf_dict[feat_filename] = np.array(mlf_dict[feat_filename])
   return mlf_dict 



def create_mapping(map_file):
   state2int = {}
   count = 1
   with open(map_file) as m:     # setup the mapping from state to integer
      for line in m:
         line = line.strip()
         if line:
            state2int[line] = count
            count += 1
   return state2int



def proc_seq(mlf_dict, feat_list, out_folder, skip_frames = 5):


   if not os.path.exists(out_folder):
      os.makedirs(out_folder)

   m, v = read_mv(stat_file)
   if m is None or v is None:
      raise Exception("mean or variance vector does not exist")


   utt_count = 0
   chunk_idx = -1
   data_cache = []
   buffer_len = 0

   f = open(feat_list, 'r')

   while True:
      if buffer_len < buffer_seq:
         line = f.readline()
         if line == '':
            print ('All utterances processed')
            f.close()
            break

         line = line.strip()
         if len(line) < 1: continue

         filename_key = line.split('/')[-1]
         label = mlf_dict[filename_key]

         io_src = htk_io.fopen(line)
         utt_feat_src = io_src.getall()
         frm_num_src, feat_dim_src = utt_feat_src.shape


         utt_feat_src -= m         # mean normalization
         utt_feat_src /= (np.sqrt(v) + eps)     # var normalization
         if skip_frames > 0:
             utt_feat_src = np.pad(utt_feat_src, ((0, skip_frames), (0,0)), mode = 'edge')    # pad the ending frames
             utt_feat_src = utt_feat_src[skip_frames:,:]    

         data_cache.append( (utt_feat_src, label) )     # fill the buffer
         buffer_len += 1

         print ("Processed %d frames for file %s" %(utt_feat_src.shape[0], filename_key))
         mlf_dict.pop(filename_key)
         utt_count += 1
         print (utt_count)
      else:     # output to hard drive
         chunk_idx += 1
         print ('Saving data chunk %d...' %chunk_idx)
         out_file = out_folder + '/' + str(chunk_idx) + '.h5'
         data_cache, buffer_len = make_chunk(out_file, data_cache, buffer_len)

   
   while buffer_len > 0:
      chunk_idx += 1
      print ('Saving remaining data chunk %d...' %chunk_idx)
      out_file = out_folder + '/' + str(chunk_idx) + '.h5'
      if buffer_len > chunk_seq:
         data_cache, buffer_len = make_chunk(out_file, data_cache, buffer_len)
      else:
         save_hd5(out_file, data_cache)
         buffer_len = 0




if __name__ == '__main__':
   mlf_dict = create_mlf_dict(mlf_file) 
   proc_seq(mlf_dict, 'train_fea.scp', 'train_ctc')
   proc_seq(mlf_dict, 'cv_fea.scp', 'cv_ctc')
