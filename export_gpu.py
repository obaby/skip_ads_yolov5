import argparse

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt',
                        help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size)).to(device='cuda')  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cuda'))  # load FP32 model

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
        ts = torch.jit.trace(model, img)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        ts = optimize_for_mobile(ts)
        ts._save_for_lite_interpreter(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)