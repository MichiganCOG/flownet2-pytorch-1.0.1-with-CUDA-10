#!/bin/bash

set -x
# change inference_dataset_root to dir to train_frames
# change s to dir where you want to save flow (fw_gan_vvt/train/optical_flow)
# Warning: do NOT change --number_gpus to above 1
python main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder \
 --inference_dataset_root /data_hdd/fw_gan_vvt/train/train_frames --log_frequency 1000 -s /data_hdd/fw_gan_vvt/train/optical_flow --number_workers 4 --resume FlowNet2_checkpoint.pth.tar --number_gpus 1

# change train_root to dir to train_frames for both commands below
python organize_flow.py --datamode train --train_root /data_hdd/fw_gan_vvt/train/train_frames
python rename_flo.py --datamode train --train_root /data_hdd/fw_gan_vvt/train/train_frames