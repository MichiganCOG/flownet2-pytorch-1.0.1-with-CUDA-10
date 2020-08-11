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
#train_list = [f for f in sorted(os.listdir(train_root)) if osp.splitext(f)[1] == '.png']
train_list = sorted(os.listdir(train_root))
flo_list = sorted(os.listdir(flow_root))
#print(train_list)
folder_names = train_list
folder_lens = []
len_dict = {}
for value in train_list:
    print(value)
    value_len = len([f for f in os.listdir(osp.join(train_root, value)) if osp.splitext(f)[1] == '.png'])
    #len_dict[value] = value_len
    folder_lens.append(value_len)
#len_dict = sorted(len_dict)
#print(type(len_dict))
#folder_names, folder_lens = len_dict.keys(), len_dict.values()
#print(len(flo_list))
#print(len(train_list))
#print(folder_lens)
#assert sum(folder_lens) == len(flo_list), f"{sum(folder_lens)} {len(flo_list)}"
#assert 1 == 0, f"{sum(folder_lens)} {len(flo_list)}"
#print(dict(zip(folder_names, folder_lens)))
for i in range(len(folder_names)):
    os.mkdir(osp.join(flow_root, folder_names[i]))
    #print(osp.join(flow_root, folder_names[i]))    
    for j in range(folder_lens[i]):
        try:
            x = flo_list.pop(0)
            print(x)
            os.system(f"mv {osp.join(flow_root, x)} {osp.join(flow_root, folder_names[i])}")
        except Exception as e:
            print(j, folder_lens[i])
            print("You can ignore this error if the above two numbers are off by 1. else, there is an issue ")
            raise e
        


