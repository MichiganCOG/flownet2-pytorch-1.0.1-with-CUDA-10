import os
import pickle
from argparse import Namespace

import numpy as np
import torch
from PIL import Image

from src.models import FlowNet2


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
WEIGHTS_PATH = os.path.join(PROJ_DIR, 'weights', 'FlowNet2_checkpoint.pth.tar')


def pil_to_tensor(image_pil):
    """Converts the given PIL Image instance to a Tensor."""
    return torch.FloatTensor(np.array(image_pil, dtype=np.float32))


def main():
    image_0 = Image.open(os.path.join(PROJ_DIR, 'images', 'image_0.png'))
    image_1 = Image.open(os.path.join(PROJ_DIR, 'images', 'image_1.png'))

    # Prepare data for forward pass
    image_0_t = pil_to_tensor(image_0)
    image_1_t = pil_to_tensor(image_1)
    data = torch.stack([image_0_t, image_1_t])  # 2 x H x W x C
    data = data.unsqueeze(0)  # 1 x 2 x H x W x C
    data = data.permute(0, 4, 1, 2, 3).contiguous()  # 1 x C x 2 x H x W
    data = data.cuda()

    # Initialize model
    args = Namespace(fp16=False, rgb_max=255.0)
    model = FlowNet2(args)
    checkpoint = torch.load(WEIGHTS_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    torch.set_grad_enabled(False)
    output = model.forward(data)  # 1 x 2 x H x W
    print(output)
    output_np = output[0].permute(1, 2, 0).cpu().numpy()  # H x W x 2
    with open('flow.pkl', 'wb') as f:
        pickle.dump(output_np, f)


if __name__ == '__main__':
    main()
