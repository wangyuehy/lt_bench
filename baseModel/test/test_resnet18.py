import base
import unittest
from base import loadNet
import pdb 

class test_resnet18(unittest.TestCase):
  def __init__(self):
    self.model = main.loadNet(task='classification',model='resnet18')

  def test_torch(self):
    net = self.model.torch(input_shape=[3,300,300], batch_size=16)
    assert 1

  def test_onnx(self):
    for inshape in [[3,224,224],[3,320,320]]:
      for batch in [-1,8,16]:
        net = model.onnx(input_shape=inshape, batch_size=batch)
 
def test_get_abs_config():
  func_name = 'torch'
  from omegaconf import OmegaConf
  from base import get_absolute_config
  base_config = OmegaConf.load('cv/classification/resnet18/resnet18.yaml')
  final_config = get_absolute_config(base_config, func_name)
  print(OmegaConf.to_yaml(final_config))

 

def test_resnet18_torch():
  model = base.loadNet(task='classification',model_type='resnet18')()
  
  net = model.torch()
  net = model.torch(return_path="build/restnet18.pth")

  
def test_resnet18_onnx():
  model = base.loadNet(task='classification',model_type='resnet18')()
  net = model.onnx()
  net = model.torch(return_path="build/restnet18.onnx")
  net = model.torch(input_shape=[3,300,300],batch_size=16)


if __name__ == '__main__':
  #test_resnet18_torch()
  test_resnet18_onnx()
  #test_get_abs_config()
