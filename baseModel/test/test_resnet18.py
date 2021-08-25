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
 


def test_resnet18_torch():
  model = base.loadNet(task='classification',model_type='resnet18')
  pdb.set_trace()
  net = model.torch()
  net = model.torch(return_path=true)
  net = model.torch(model_path='xxx.pth')
  net = model.torch(input_shape=[3,300,300],batch_size=16)
  
def test_resnet18_onnx():
  model = base.loadNet(task='classification',model='resnet18')
  pdb.set_trace()
  net = model.onnnx()
  net = model.torch(return_path=true)
  net = model.torch(input_shape=[3,300,300],batch_size=16)


if __name__ == '__main__':
  test_resnet18_torch()

