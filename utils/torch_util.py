"""
PyTorch related util functions
"""
import torch

def allocate_gpu(id=None):
    '''
    choose the free gpu in the node
    '''
    v = torch.empty(1)
    if id is not None:
        return torch.device("cuda:{}".format(str(id)))
    else:
        for i in range(8):
            try:
                dev_name = "cuda:{}".format(str(i))
                v = v.to(dev_name)
                print("Allocating cuda:{}.".format(i))

                return v.device
            except Exception as e:
                pass
        print("CUDA error: all CUDA-capable devices are busy or unavailable")
        return v.device

