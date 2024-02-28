import numpy as np
from PIL import Image
import random
from torchvision.transforms import InterpolationMode
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
# some transform will be used in referring image segmentation
# before to tensor, img is Pillow Image Object
def pad_if_smaller(img,size,fill=0):
    min_size=min(img.size)
    if min_size < size:
        ow,oh=img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img=F.pad(img,(0,0,padw,padh),fill=fill)
    
    return img

class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms
    
    def __call__(self, image,target):
        for t in self.transforms:
            image,target=t(image,target)
        
        return image,target

# 随机resize
class RandomResize(object):
    def __init__(self,min_size,max_size=None):
        self.min_size=min_size
        if max_size is None:
            max_size=min_size
        self.max_size=max_size
    
    def __call__(self,image,target):
        size=random.randint(self.min_size,self.max_size)
        image=F.resize(image,size)
        target=F.resize(target,size,interpolation=InterpolationMode.NEAREST)
        return image,target

# 固定resize

class Resize(object):
    def __init__(self,output_size=384,train=True) -> None:
        self.size=output_size
        self.train=train
    def __call__(self, image,target):
        image=F.resize(image,(self.size,self.size))
        # we must need to test on the original size 
        if self.train:
            target=F.resize(target,(self.size,self.size),interpolation=InterpolationMode.NEAREST)
        return image,target



class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target



class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target.copy()), dtype=torch.int64)
        return image, target

class RandomAffine(object):
    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, image, target):
        affine_params = T.RandomAffine.get_params(self.angle, self.translate, self.scale, self.shear, image.size)
        image = F.affine(image, *affine_params)
        target = F.affine(target, *affine_params)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# We don't apply other complex data argumentation
def get_transform(args):
    transforms = []
    # transforms.append(Resize(args.size,not args.eval))
    transforms.append(Resize(args.size, args.size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)