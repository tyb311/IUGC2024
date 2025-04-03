import pickle, os, socket
import numpy as np

import importlib, subprocess, sys
def install_if_missing(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        print(f'Package {package_name} is  missing! installing now')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

print('#'*4, 'Installing mising packages')
# install_if_missing('glob')
hostname = socket.gethostname()
print('hostname:', hostname)
if hostname!='Tan':
    install_if_missing('torch==1.10.1')
print('#'*4, 'Importing mising packages')
# from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
print('PACKAGE:torch', torch.__version__)



# Defing model for validation
class model:
    def __init__(self):
        '''
        interface class 
        '''
        self.mean = None
        self.std = None
        print('#'*4, 'Defing model')
        

    def load(self, path="./"):
        print('#'*4, 'Loading model:', path, os.path.isfile(path))
        folder = path.replace('\\', '/')
        filename = folder.split('/')[-1]
        folder = folder.replace(filename, '')
        # print('Files submitted:', glob(folder+'*'))
        # print('DATA Files1:', glob('/tmp/codalab/tmpLEphbm/run/input/input_data_1_2/*'))
        # print('DATA Files2:', glob('/tmp/codalab/tmpLEphbm/run/input/*'))
        # print('RUN Files:', glob('/tmp/codalab/tmpLEphbm/run/*'))
        """_summary_
        your model weight or pickle file name should use model.pickle ...
        """
        # file = open(path, "rb")
        # self.model = pickle.load(file)

        # 加载并测试TorchScript模型  
        example_input = torch.rand(1,3,256,256)
        print('Traced Model:', folder+"traced_model.pt")
        self.model = torch.jit.load(folder+"traced_model.pt")  
        out_cls, out_seg = self.model(example_input)  
        print('Dummy Test:', out_cls.shape, out_seg.shape)
        
        # print('#'*4, 'Loaded model:', path)
        return self

    def predict(self, X, isLabeled=True):
        print('#'*4, 'Predicting model')
        """
        X: numpy array of shape (3,512,512)
        isLabeled (bool): we only label one image in a video. True represents this frame is labeled.
        """
        self.model.eval()
        image = torch.tensor(X, dtype=torch.float).unsqueeze(0)/255.0  # (1,3,512,512)
        image = F.interpolate(image, size=(256,256), mode='bilinear', align_corners=False)
        out_cls, seg = self.model(image)  # cls:(1,2)  seg:(1,3,512,512)  seg_flag: based on classification result by model
        seg = F.interpolate(seg, size=(512,512), mode='bilinear', align_corners=False)
        
        # seg_flag = torch.softmax(out_cls, dim=-1)
        # seg_flag = isLabeled and seg_flag[0][1].item() >= 0.5
        # # if not seg_flag:
        # #     seg = None
        
        seg_flag = torch.softmax(out_cls, dim=-1)
        seg_flag = seg_flag[0][1].item() >= 0.5
        if not isLabeled:
            seg = None
            
        if seg is not None:
            seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (512,512,3)
            """
                postprocess should be execute on segmentation
                you should design your own postprocess algorithm
            """

        return out_cls.detach().numpy(), seg, seg_flag

    def save(self, path="./"):
        print('#'*4, 'Saving model')
        '''
        Save a trained model.
        '''
        pass




# Defing model for test
class modelInter:
    def __init__(self):
        '''
        interface class 
        '''
        self.mean = None
        self.std = None
        print('#'*4, 'Defing model')
        

    def load(self, path="./"):
        print('#'*4, 'Loading model:', path, os.path.isfile(path))
        folder = path.replace('\\', '/')
        filename = folder.split('/')[-1]
        folder = folder.replace(filename, '')
        # print('Files submitted:', glob(folder+'*'))
        # print('DATA Files1:', glob('/tmp/codalab/tmpLEphbm/run/input/input_data_1_2/*'))
        # print('DATA Files2:', glob('/tmp/codalab/tmpLEphbm/run/input/*'))
        # print('RUN Files:', glob('/tmp/codalab/tmpLEphbm/run/*'))
        """_summary_
        your model weight or pickle file name should use model.pickle ...
        """
        # file = open(path, "rb")
        # self.model = pickle.load(file)

        # 加载并测试TorchScript模型  
        example_input = torch.rand(1,3,256,256)
        print('Traced Model:', folder+"traced_model.pt")
        self.model = torch.jit.load(folder+"traced_model.pt")  
        out_cls, out_seg = self.model(example_input)  
        print('Dummy Test:', out_cls.shape, out_seg.shape)
        
        # print('#'*4, 'Loaded model:', path)
        return self

    def predict(self, X, isLabeled=True):
        print('#'*4, 'Predicting model')
        """
        X: numpy array of shape (3,512,512)
        isLabeled (bool): we only label one image in a video. True represents this frame is labeled.
        """
        self.model.eval()
        image = torch.tensor(X, dtype=torch.float).unsqueeze(0)/255.0  # (1,3,512,512)
        image = F.interpolate(image, size=(256,256), mode='bilinear', align_corners=False)
        out_cls, seg = self.model(image)  # cls:(1,2)  seg:(1,3,512,512)  seg_flag: based on classification result by model
        seg = F.interpolate(seg, size=(512,512), mode='bilinear', align_corners=False)
        
        seg_flag = torch.softmax(out_cls, dim=-1)
        seg_flag = seg_flag[0][1].item() >= 0.5
        if not isLabeled:
            seg = None
        
        if seg is not None:
            seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (512,512,3)
            """
                postprocess should be execute on segmentation
                you should design your own postprocess algorithm
            """

        return out_cls.detach().numpy(), seg, seg_flag

    def save(self, path="./"):
        print('#'*4, 'Saving model')
        '''
        Save a trained model.
        '''
        pass


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    def imread(path_imag):
        img = Image.open(path_imag).convert('RGB').resize((512,512))
        return np.array(img)


    if os.path.isdir(r'G:\Objects\Challenges2024\IUGC'):
        raw = imread(r'G:\Objects\Challenges2024\IUGC\iugc_cls\test\pos/CASE_1427.png')
        # raw = imread(r'G:\Objects\Challenges2024\IUGC\iugc_cls\test\neg/CASE_1373.png')
    else:
        raw = imread(r'pos/CASE_0013.png')
    print(raw.shape)

    img = np.transpose(raw, axes=[2,0,1])
    print(img.shape)
    
    # net = model()
    net = modelInter()
    try:
        net.load(r'submit_code\iris_model.pickle')
    except:
        net.load(r'iris_model.pickle')
    y,k,f = net.predict(img)

    print(y.shape, y, f)
    if k is not None:
        print('seg:', k.shape)

        seg = k#.data.numpy()
        plt.subplot(121), plt.imshow(raw)
        plt.subplot(122), plt.imshow(seg)
        plt.show()

    '''
    Output:
        (512, 512, 3)
        (3, 512, 512)
        (1, 2) (512, 512) False
    '''
