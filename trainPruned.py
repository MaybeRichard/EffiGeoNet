import torch
from modelNetMDistill import GeoNetM,EPELoss


import torch.nn.functional as F
from torch.optim import SGD

# device = torch.device('cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu')
device = "cpu"
geoNet = GeoNetM().to(device)
import numpy as np
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
print(count_parameters(geoNet))

optimizer = SGD(geoNet.parameters(),1e-2)
criterion = F.nll_loss

config = [{
    'sparsity_per_layer':0.5,
    'op_types':['Conv2d',"Linear"]
}]

from nni.compression.pytorch.pruning import L2NormPruner
pruner = L2NormPruner(geoNet,config)
_,masks = pruner.compress()
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

import matplotlib.pyplot as plt
for _,mask in masks.items():
    mask = mask['weight'].detach().cpu().numpy()
print("sparsity:{}".format(mask.sum()/mask.size))
# need to unwrap the model, if the model is wrapped before speedup
pruner._unwrap_model()

# speedup the model
from nni.compression.pytorch.speedup import ModelSpeedup

ModelSpeedup(geoNet, torch.rand(1, 3, 512, 512).to(device), masks).speedup_model()
print(count_parameters(geoNet))

import time

def test_model_speed(model, input_data):
    # 把模型放到评估模式
    model.eval()

    # 把输入数据移到相应的设备
    input_data = input_data.to(device)

    # 确保 CUDA 操作同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 记录开始时间
    start_time = time.time()

    # 执行前向传播
    with torch.no_grad():
        _ = model(input_data)

    # 确保 CUDA 操作同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 记录结束时间
    end_time = time.time()

    # 计算并返回模型的推理时间
    return end_time - start_time


# 创建一个随机输入数据
input_data = torch.rand(1, 3, 512, 512)

# 测试模型的推理速度
inference_time = test_model_speed(geoNet, input_data)

print(f'Inference time: {inference_time} seconds')
