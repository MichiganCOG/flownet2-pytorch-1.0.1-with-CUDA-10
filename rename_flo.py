import os
import os.path as osp
from IPython import embed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datamode", type=str, default="train")
parser.add_argument("--train_root", type=str, default="/data_hdd/fw_gan_vvt/train/train_frames")
opt = parser.parse_args()
train_root = opt.train_root
flow_root = osp.join(train_root.replace(f"{opt.datamode}_frames", "optical_flow"), "inference", "run.epoch-0-flow-field")
assert osp.exists(train_root)
assert osp.exists(flow_root)
train_list = sorted(os.listdir(train_root))
flo_list = sorted(os.listdir(flow_root))
folder_names = train_list
folder_lens = []
len_dict = {}
for value in train_list:
    os.chdir(osp.join(flow_root, value))
    #print(os.listdir('.'))
    
    rename = "_".join(os.listdir(osp.join(train_root, value))[0].split("_")[:2])

    for i, flo in enumerate(sorted(os.listdir('.'))):
        new = f"{rename}_{i+1:03}.flo"
        #assert 1 == 0, print(flo, new)
        print(flo, new)
        os.rename(flo, new)
    #print(value)
    #value_len = len(os.listdir(osp.join(train_root, value)))
    #len_dict[value] = value_len
    #folder_lens.append(value_len)
#embed()
