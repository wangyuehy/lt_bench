import main

def test_resnet18():
  resmodel = main.loadNet(task='classification', model_type='resnet18')

def test_resnet18_torch():
  resmodel = main.loadNet(task='classification', model_type='resnet18')
  resmodel.torch()

  