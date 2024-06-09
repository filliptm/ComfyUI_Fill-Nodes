import torch
h,w = inputs[0].shape[:2]
b = torch.tensor([0.2, 0.8, 0.5]).view(1, 1, 3).expand(h, w, 3)
outputs[0] = (b + inputs[0]) * 0.5