# Imports for transform and dataset prepration

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

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

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
  return x[:32].float()
class shift():
  def __init__(self, sz):
    self.sz = sz
  def __call__(self, x):
    return torch.permute(x, self.sz)
def pr(x):
  print(x.shape)
  return x
def vid_transform(t):
  return transforms.Compose([ApplyTransformToKey(key="video", transform = t)])
train_transform = transforms.Compose([
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.CenterCrop(224),
                    transforms.Resize(224),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
                  ])

# Load Dataset

batch_size = 8
def collate_fn(batch):
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    y = [int(data_item[2]) for data_item in batch]
    return x[:32], y

train_kinetics = datasets.Kinetics("../k400val_pytorch", frames_per_clip= 32, split='val', num_classes= '400', step_between_clips= 2, transform = train_transform,  download= False, num_download_workers= 1, num_workers= 80)
#train_ucf = datasets.UCF101("./data/", annotation_path = "ucf_annotation.csv", frames_per_clip= 32, step_between_clips = 2, transform = train_transform, num_workers= 10)
#train_hmdb51 = datasets.HMDB51("./data/", annotation_path = "hmdb51_annotation.csv", frames_per_clip= 32, step_between_clips = 2, transform = train_transform, num_workers= 10)
#train_ds = torch.utils.data.ConcatDataset([train_kinetics, train_ucf, train_hmdb51])
train_ds = train_kinetics
test_ds = train_kinetics
train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size = batch_size, shuffle = True);
test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = True);

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

def train_with_extraction(model, victim):
    # again, batch_size=1 due to compute restrictions on colab
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    criterion = nn.CrossEntropyLoss()

    for idx in range(10):
      print('\nStarting Epoch: {}\n'.format(idx))
      rloss = 0.0;
      model.train()
      for step,(video, label) in enumerate(train_dl):
          torch.cuda.empty_cache()
          optimizer.zero_grad()
          video = Variable(video.to(DEVICE), requires_grad=False)
          video = video.permute(0, 2, 1, 3, 4)
          pred = model(video)
          label_ = torch.argmax(victim(video), dim=1)
          loss = criterion(pred,label_)
          rloss+=loss.item()
          loss.backward()
          optimizer.step()
          # print(f'Predicted class: {torch.argmax(pred, dim=1)}, Teacher class: {label_}, Actual label: {label}')
          print(rloss/(step+1))

      print(f'avg loss: {rloss/len(train_dl)}')
      print('evaluation:')
      model.eval()
      with torch.no_grad():
        for s,(v,l) in enumerate(test_dl):
          v = v.permute(0, 2, 1, 3, 4)
          p = model(v)
          l_ = victim(v)
          # print(f'Predicted class: {torch.argmax(p, dim=1).item()}, Teacher class: {l_.item()}, Actual label: {l}')
          print(f'Accuracy : {(torch.argmax(p, dim=1) == l).float().mean()}')

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

    adversary = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)

    # model_adversary = build_model(cfg.model, train_cfg=None, test_cfg=None)
    adversary.to(DEVICE)
    # adversary = MMActionModelWrapper(model_adversary)
    victim.eval()
    print(cfg.model)
    train_with_extraction(adversary, victim)
