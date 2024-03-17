import numpy as np

from PIL import Image
import torch
import torch.nn as nn
from modelNetM import EncoderNet, DecoderNet, ClassNet
from torchvision import transforms
from pathlib import Path
def load_weights(model, path):
    # model = nn.DataParallel(model) # if pre-trained
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(torch.load(path))
    return model.eval()

def transform_image(image_path):
    transform = transforms.Compose([transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])
    im = Image.open(image_path).convert('RGB')
    im_npy = np.asarray(im.resize((512,512)))
    im_tensor = transform(im).unsqueeze(0)
    if torch.cuda.is_available(): im_tensor = im_tensor.cuda()
    return im_tensor,im_npy

def rectify_image(image_path, multi=1):
    from resampling import rectification
    im_tensor,im_npy = transform_image(image_path)
    middle = model_en(im_tensor)
    flow_output = model_de(middle)
    return rectification(im_npy, flow_output.data.cpu().numpy()[0]*multi)

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()

model_en = load_weights(model_en, './model_en_30.pkl')
model_de = load_weights(model_de, './model_de_last.pkl')
model_class = load_weights(model_class, './model_class_30.pkl')

testImgPath = './dataset/moving/test_distorted/'
testImgs = [x for x in [Path(testImgPath).rglob(e) for e in ('*.jpg','*.png')] for x in x]


imgPath = testImgs[3]
print(testImgs)
resImg,resMsk = rectify_image(imgPath)
Image.fromarray(resImg).save('resImg.png')