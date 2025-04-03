from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch, pickle, os
import numpy as np
print('torch:', torch.__version__)

import segmentation_models_pytorch as smp
class SmpNet(nn.Module):
    __name__='SmpNet'
    def __init__(self, in_channels=1, out_channels=3, name='vnet', encoder='efficientnet-b3'):
        super(SmpNet, self).__init__()
        #encoders:efficientnet-b5,xception,densenet169,densenet121,timm-skresnet34,se_resnet50,timm-regnety_040,timm-regnety_032
        # timm-res2next50,timm-resnest50d,resnext50_32x4d,resnet50
        weights = None#'imagenet'
        encoder = 'resnet50'
        encoder = 'efficientnet-b3'
        encoder = 'timm-regnety_040'
        
        base = smp.DeepLabV3(encoder_name=encoder, encoder_depth=5, encoder_weights=weights, decoder_channels=128, upsampling=8, 
                in_channels=in_channels, classes=out_channels, activation=None, aux_params=None)

        self.base = base
        self.__name__ = name
    
    def tta(self, imag1):
        # print('tta-seg')
        
        imag2 = torch.flip(imag1, dims=(-1,))
        imag3 = torch.flip(imag1, dims=(-2,))
        imag4 = torch.flip(imag1, dims=(-1,-2,))

        image = torch.concat([imag1,imag2,imag3,imag4], dim=0)
        with torch.no_grad():
            output = self.base(image)
        imag1,imag2,imag3,imag4 = torch.chunk(output, chunks=4, dim=0)
        imag2 = torch.flip(imag2, dims=(-1,))
        imag3 = torch.flip(imag3, dims=(-2,))
        imag4 = torch.flip(imag4, dims=(-1,-2,))
        
        pred = torch.concat([imag1,imag2,imag3,imag4], dim=0).mean(dim=0).unsqueeze(0)
        return pred

    def forward(self, x):
        x = x[:,:1]
        
        x = x-x.mean()
        x = x/(x.std()+1e-6)

        # x = x*255.0
        # x = x-42.23065150483229
        # x = x/42.27579027160423

        if self.training:
            x = self.base(x)
        else:
            x = self.tta(x)
        return x


def smpdv3(**args):
    return SmpNet(name='smpdv3', **args)

import timm, torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
class VisionNetV2(nn.Module):
    __name__ = 'tvn'
    NETS = [				
        'cnext_s','cnext_t','cnext_b',
        'eff_v2l','eff_v2m','eff_v2s',
        'regnet_y','regnet_x',
        'alex','vgg11',
        'res18','res34','res50','res50k',
        'maxvit_t','swin_t','swin_s','swin_v2_t','swin_v2_s',
        'resnext34','resnext50','shufflenet','squeezenet','googlenet','mnasnet','mobilenet_v2''mobilenet_v3','vit16','vit32',
        'vit','mobilenet_v2','mobilenet_v3',
        'swin_t','swin_s','swin_v2_t','swin_v2_s',
    ]
    def __init__(self, base, flag224=False, nb_classes=5):
        super(VisionNetV2, self).__init__()
        self.base = base
        self.flag224 = flag224
        # self.base = svf(base)
        # self.fc1 = fc1
        # print('base.DIM_DENSE_LAYER:', base.DIM_DENSE_LAYER)
        # self.fds = FDS(
        # 		feature_dim=base.DIM_DENSE_LAYER, bucket_num=100, bucket_start=3,
        # 		start_update=0, start_smooth=1, kernel='gaussian', ks=9, sigma=1, momentum=0.9
        # 	)

    def forward(self, x):
        # print(x.shape)
        x = self.base(x)

        return x#self.fc1(x)


def load_cls_model(name, num_classes=2, in_channels=3):
    nb_classes = num_classes
    if 'squeezenet' in name:
        net = models.squeezenet1_1(num_classes=num_classes)
        net = VisionNetV2(net, flag224='vit' in name, nb_classes=nb_classes)
    elif 'shufflenet' in name:
        net = models.shufflenet_v2_x1_0(num_classes=num_classes)
        net = VisionNetV2(net, flag224='vit' in name, nb_classes=nb_classes)
    elif 'effb3' in name:#10.7MB
        net = timm.create_model('tf_efficientnet_b3', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    elif name == 'pool':
        net = timm.create_model('poolformerv2_s12', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    elif name == 'lcnet':
        net = timm.create_model('lcnet_100', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    elif name == 'res34':
        net = timm.create_model('resnet34', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    elif name == 'regnetx':
        net = timm.create_model('regnetx_002', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    elif name == 'dense121':
        net = timm.create_model('densenet121', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    elif name == 'regnetx':
        net = timm.create_model('regnetx_002', pretrained=False, num_classes=nb_classes, in_chans=in_channels)
    return net

class IUGCNet(nn.Module):
    def __init__(self, mode="all",
        path_seg =r'G:\Objects\Challenges2024\IUGC\checkpoints/deeplabv3.pt',
        
        path_cls7=r'G:\Objects\Challenges2024\IUGC\checkpoints/v2/lcnet.pt',
        path_cls8=r'G:\Objects\Challenges2024\IUGC\checkpoints/v2/regnetx.pt',
        path_cls9=r'G:\Objects\Challenges2024\IUGC\checkpoints/v2/res34.pt',
        path_cls0=r'G:\Objects\Challenges2024\IUGC\checkpoints/v2/effb3.pt',
        path_clsa=r'G:\Objects\Challenges2024\IUGC\checkpoints/v2/dense121.pt',
        path_clsb=r'G:\Objects\Challenges2024\IUGC\checkpoints/v2/shufflenet.pt',
        ) -> None:
        super().__init__()
        print('$'*32, 'IUGCNet')
        self.seg_fcn = self.load(smpdv3(), path_seg)

        self.nets = nn.ModuleList([
            # self.load(load_cls_model('dense121'), path_clsa),   #x
            # self.load(load_cls_model('shufflenet'), path_clsb), #x

            # self.load(load_cls_model('lcnet'), path_cls7),      #x
            # self.load(load_cls_model('regnetx'), path_cls8),    #x
            self.load(load_cls_model('res34'), path_cls9, flag_base=True),
            # self.load(load_cls_model('effb3'), path_cls0, flag_base=True),
        ])

    def load(self, net, path_ckpt, flag_model=True, flag_base=False):
        if not os.path.exists(path_ckpt):
            return net 

        ckpt = torch.load(path_ckpt, map_location='cpu')
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        
        pth_new = {}#OrderedDict()
        for key,itm in ckpt.items():
            if flag_model and key.startswith('model.'):
                key = key.replace('model.',  '')
            if flag_base and key.startswith('base.'):
                key = key.replace('base.',  '')
            pth_new[key] = itm

        print('loading:', net.load_state_dict(pth_new, strict=True), path_ckpt)        
        net.eval()
        # for p in net.parameters():
        #     p.requires_grad = False
        return net

    def tta(self, net, imag1):
        imag2 = torch.flip(imag1, dims=(-1,))
        # imag3 = torch.flip(imag1, dims=(-2,))
        # imag4 = torch.flip(imag1, dims=(-1,-2,))

        image = torch.concat([imag1,imag2], dim=0)
        with torch.no_grad():
            output = net(image)
        # print('tta-cls', output)
        # output = output.softmax(dim=-1)
        # imag1,imag2 = torch.chunk(output, chunks=2, dim=0)
        # imag1 = imag1 / (1e-5+imag1.std())
        # imag2 = imag2 / (1e-5+imag2.std())
        # output = torch.concat([imag1,imag2], dim=0)
        output = output.mean(dim=0).unsqueeze(0)
        return output

    def forward(self, inp, mode="inference"):
        if mode == "inference":
            modes = ["seg", "cls"]
        else:
            modes = [mode]

        if "seg" in modes:
            out_seg = self.seg_fcn(inp)

        if "cls" in modes:
            out = 0
            for net in self.nets:
                # out = out + net(inp)
                # print(out)
                if self.training:
                    out = out + net(inp)
                else:
                    out = out + self.tta(net, inp)
            out_cls = out / len(self.nets)
            
            # seg_flag = torch.softmax(out_cls, dim=-1)
            # seg_flag = seg_flag[0][1].item() >= 0.5
            # if not seg_flag:
            #     out_seg = None

        # print('IUGCNet:', inp.shape, seg.shape)
        if mode == "seg":
            return out_seg
        elif mode == "cls":
            return out_cls
        else:
            return out_cls, out_seg




if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt
    def imread(path_imag):
        img = Image.open(path_imag).convert('RGB').resize((512,512))
        return np.array(img)

    # import pickle

    # with open('model.pickle', 'wb') as f:
    #     pickle.dump(model, f)

    net = IUGCNet()
    net.eval()

    # with open('submit_code/model.pickle', 'wb') as f:
    #     pickle.dump(net, f)

    # raw = imread(r'G:\Objects\nnUNet_raw\Dataset030_IUGC\imagesTs/CASE_3201_0000.png')
    raw = imread(r'G:\Objects\Challenges2024\IUGC\iugc_cls\test\pos/CASE_1427.png')
    # raw = imread(r'G:\Objects\Challenges2024\IUGC\iugc_cls\test\neg/CASE_1373.png')
    
    img = np.transpose(raw, axes=[2,0,1])
    img = torch.from_numpy(img).float().unsqueeze(0)/255
    print(img.shape)
    
    y,k = net(img)

    print(y.shape, y.argmax(dim=-1), y.reshape(-1))
    if k is not None:
        print('seg:', k.shape)


    example_input = img

    # 使用torch.jit.trace将模型转换为TorchScript  
    traced_script_module = torch.jit.trace(net, example_input)  
    traced_script_module.save("submit_code/traced_model.pt")  
    
    # # 或者，如果你希望使用torch.jit.script（可能需要对模型进行一些修改）  
    # scripted_module = torch.jit.script(net)  
    # scripted_module.save("submit_code/traced_model.pt")  
    
    # 加载并测试TorchScript模型  
    loaded_model = torch.jit.load("submit_code/traced_model.pt")  
    output = loaded_model(example_input)  
    print(output[0].shape, output[1].shape)


    ####################################################################################
    #   基于ONNX，失败于Upsampling-Bilinear算子
    ####################################################################################
    # # 假设输入是一个单通道的28x28的图像，batch size为1  
    # dummy_input = img
    # # 使用torch.onnx.export()函数导出模型  
    # onnx_path = 'submit_code/model.onnx'  
    # torch.onnx.export(net,               # 训练的模型  
    #                 dummy_input,         # 模型输入 (或是一个tuple用于多输入)  
    #                 onnx_path,           # 输出的ONNX文件名  
    #                 export_params=True,  # 存储训练好的参数权重  
    #                 opset_version=10,    # ONNX版本  
    #                 do_constant_folding=True,  # 是否执行常量折叠优化  
    #                 input_names = ['input'],   # 输入节点的名称  
    #                 output_names = ['output_cls', 'output_seg'], # 输出节点的名称  
    #                 dynamic_axes={
    #                     'input' : {0 : 'batch_size'},    # 声明batch_size是动态变化的  
    #                     'output_cls' : {0 : 'batch_size'},
    #                     'output_seg' : {0 : 'batch_size'},
    #                     })  
    # print(f'Model has been exported to {onnx_path}')
    ####################################################################################



