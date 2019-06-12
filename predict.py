import torch
import numpy as np
from logistic_pytorch import Net

if __name__ == "__main__":
    length = np.load("length.npy").tolist()
    print(length)

    net = Net(sum(length),64,1).cuda()
    net.load_state_dict(torch.load("net.pt"))