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
  net = model.torch(input_shape=[3,300,300],batch_size=16, return_path='build/resnet18_3300300.onnx')

class test_resnet18:
  def __init__(self):
    self.model = base.loadNet(task='classification', model_type='resnet18')()

  def test_onnx(self):
    # default used model zoo
    net = self.model.onnx()
    # different shape, call build onnx
    net = self.model.onnx(input_shape=[3,300,300],batch_size=8,return_path='build/3x300x300_b8_resent18.onnx')
    # same shape with model zoo, cp from model zoo
    net = self.model.onnx(input_shape=[3,224,224],return_path='build/3x224x224_b8_resent18.onnx')

  def test_build_onnx(self):
    net = self.model.build_onnx()

  def test_build_trt_engine(self):
    net = self.model.build_trt_engine()
    net = self.model.build_trt_engine(trt_engine_path='build/resnet18_trt_test_int8.engine', input_shape=(3,320,320),precision='int8')
    net = self.model.build_trt_engine(trt_engine_path='build/resnet18_trt_test_fp16.engine', input_shape=(3,320,320),precision='fp16')
  
  def test_prerpocess(self):
    
    self.model.preprocess(source_dir='build/preprocess/src',dst_dir='build/preprocess/dst1')
    self.model.preprocess(source_dir='build/preprocess/src',source_file='build/preprocess/valmap.txt')
    #self.model.preprocess(source_dir='build/preprocess/src')
if __name__ == '__main__':
  #test_resnet18_torch()
  #test_resnet18_onnx()
  #test_resnet18().test_onnx()
  #test_resnet18().test_build_onnx()
  #test_resnet18().test_build_trt()
  test_resnet18().test_prerpocess()
  #test_get_abs_config()
