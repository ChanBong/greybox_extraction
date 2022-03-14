# Imports for transform and dataset prepration
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch.nn.functional as func
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable

from movinets import MoViNet
from movinets.config import _C
from c3d_pytorch.C3D_model import C3D

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

def topk(output, target, maxk=5):
    """Computes the precision@k for the specified value of maxk"""
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:maxk].view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)


# Train transform and other utils

from torchvision.transforms.transforms import ToTensor
data_flip = transforms.Compose([
    transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.Normalize(0,1)
     #AddGaussianNoise(0., 10/255)
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.RandomRotation(30),
    transforms.Normalize(0,1)
])

test_transform = transforms.Compose([
     transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.Normalize(0,1)
])


# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.Normalize(0,1)
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.Normalize(0,1)
])

# Resize, normalize and crop image
data_center = transforms.Compose([
	  transforms.Resize((36, 36)),
    transforms.CenterCrop(28),
     transforms.ToTensor(),
    transforms.Normalize(0,1)
])
# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
	  transforms.Resize((28, 28)),
     transforms.ToTensor(),
    transforms.ColorJitter(brightness=5),
    transforms.Normalize(0,1)
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
	  transforms.Resize((28, 28)),
     transforms.ToTensor(),
    transforms.ColorJitter(saturation=5),
    transforms.Normalize(0,1)
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
	  transforms.Resize((28, 28)),
     transforms.ToTensor(),
    transforms.ColorJitter(contrast=5),
    transforms.Normalize(0,1)
    ])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.ColorJitter(hue=0.4),
    transforms.Normalize(0,1)
])

def tofloat(x):
  # return x[:32].float()
    return x.float()
class shift():
  def __init__(self, sz):
    self.sz = sz
  def __call__(self, x):
    return torch.permute(x, self.sz)
def pr(x):
  print(x.shape)
  return x

train_transform = transforms.Compose([
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.CenterCrop(224),
                    transforms.Resize(224),
                    transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.ColorJitter(brightness=2),
                    # transforms.RandomRotation(15)
                  ])


# Load Dataset

batch_size = 16
def collate_fn(batch):
    # print(batch[:10])
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    y = [int(data_item[2]) for data_item in batch]
    # return x[:32], y
    return x, y
    

train_kinetics = datasets.Kinetics("../k400val_pytorch_dummy", frames_per_clip= 16, split='val', num_classes= '400', step_between_clips= 2000000, transform = train_transform,  download= False, num_download_workers= 1, num_workers= 80)
#train_ucf = datasets.UCF101("./data/", annotation_path = "ucf_annotation.csv", frames_per_clip= 32, step_between_clips = 2, transform = train_transform, num_workers= 10)
#train_hmdb51 = datasets.HMDB51("./data/", annotation_path = "hmdb51_annotation.csv", frames_per_clip= 32, step_between_clips = 2, transform = train_transform, num_workers= 10)
#train_ds = torch.utils.data.ConcatDataset([train_kinetics, train_ucf, train_hmdb51])
train_ds = train_kinetics
test_ds = train_kinetics
train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size = batch_size, shuffle = True);
test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size = batch_size, shuffle = True);


# Main code for network extraction 


torch.cuda.empty_cache()
import warnings
warnings.filterwarnings('ignore')
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmaction.models import build_model
from mmcv import Config, DictAction

config = 'configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py'
checkpoint = '../swin_base_patch244_window877_kinetics400_1k.pth'

DEVICE = 'cuda:0'
SPATIAL_DIM = 224
TEMPORAL_DIM = 16
NUM_CHANNELS = 3

def get_accuracy(pred, actual):
    correct_labels = 0
    # print(f'length of pred {len(pred)}', pred, actual)
    for i in range(len(pred)):
        if (pred[i]==actual[i]):
            correct_labels+=1
    return (correct_labels/len(pred))*100.0


def train_with_extraction(model, victim):
    # again, batch_size=1 due to compute restrictions on colab
    ct = 0
    ls1 = []
    ls2 = []
    for child in model.children():
        ct += 1
        if ct >= 14:
           ls2+=list(child.parameters())
        else:
           ls1+=list(child.parameters())
    optim1 = torch.optim.Adam(ls1, lr=0.001)
    optim2 = torch.optim.Adam(ls2, lr=0.01)
    criterion = nn.MSELoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #criterion = nn.MSELoss()

    for idx in range(10):
      print('\nStarting Epoch: {}\n'.format(idx))
      rloss = 0.0;
      model.train()
      for step,(video, label) in enumerate(train_dl):
          if step>1:
              break
          torch.cuda.empty_cache()

          optim1.zero_grad()
          optim2.zero_grad()
          
          video = Variable(video.to(DEVICE), requires_grad=False)
          video = video.permute(0, 2, 1, 3, 4)
          label_ = victim(video)
         
          video = size_changer(video)
          pred = model(video)

          # print(pred.size(), label_.size())
          
          loss = criterion(pred,label_)
          rloss+=loss.item()
          loss.backward()
          optim1.step()
          optim2.step()
          # print(f'Predicted class: {torch.argmax(pred, dim=1)}, Teacher class: {label_}, Actual label: {label}')
          print(rloss/(step+1))

      print(f'avg loss: {rloss/len(train_dl)}')
      print('evaluation:')
      model.eval()
      with torch.no_grad():
          acc = []
          for step,(video, label) in enumerate(test_dl):
              video = Variable(video.to(DEVICE), requires_grad=False)
              video = video.permute(0, 2, 1, 3, 4)
              l_ = victim(video)
              video = size_changer(video)
              prediction = model(video)
          # l_ = victim(video)
              print(f'Predicted class: {torch.argmax(prediction, dim=1)}, Teacher class: {torch.argmax(l_, dim=1)}, Actual label: {label}')
          # print(torch.argmax(prediction, dim=1), label)
          # print(f'Accuracy : {(torch.sum(torch.argmax(prediction, dim=1) == label)/len(label))*100.0}%')
              print(f'Accuracy : {get_accuracy(torch.argmax(prediction, dim=1).tolist(), label)}')
              acc.append(topk(prediction, torch.argmax(l_, dim=1), 1))
      print(np.mean(acc))

# wrapper around mmactions Recognizer3d class to provide nn.Module like interface
# (for compatibility with ignite methods)
class MMActionModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model.forward_dummy(X)[0]

cfg = Config.fromfile(config)
if __name__ == '__main__':
    model_victim = build_model(cfg.model, train_cfg=None, test_cfg=None)
    # loading pretrained weights to victim
    load_checkpoint(model_victim, checkpoint, map_location=DEVICE)
    model_victim.to(DEVICE)
    victim = MMActionModelWrapper(model_victim)
    for param in victim.parameters():
      param.requires_grad = False
    victim.eval()

    net = C3D()
    net.load_state_dict(torch.load('c3d.pickle'))
    net.fc8 = nn.Linear(in_features=4096, out_features=400, bias=True)
    size_changer = torch.nn.AvgPool3d((1, 2, 2), stride=None, padding=0, ceil_mode=False)
    # adversary = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)

    # model_adversary = build_model(cfg.model, train_cfg=None, test_cfg=None)
    
    adversary = net
    adversary.to(DEVICE)
    
    # adversary = MMActionModelWrapper(model_adversary)
    # victim.eval()
    # print(cfg.model)

    train_with_extraction(adversary, victim)
