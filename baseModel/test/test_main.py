
import base
from base import loadNet
def test_loadNet():
  
  model = loadNet(task="object_detection", model="yolov3", backbone="resnet18")


def test_infer():
  model = loadNet(task="object_detection", model_type="yolov3", backbone="resnet18")
  # 1  with custom onnx model
  model.infer(modelpath='', type='onnx')

  # 2 with default onnx model
  model.infer(type='onnx')

  # 3 wtih default trt
  model.infer(type='trt')

  # 4 with custom trt
  model.infer(model_path='',type='trt')

def test_buildengine():
  model = loadNet(task="object_detection", model_type="yolov3", backbone="resnet18")
  model.preprocess()

  # 1. return officially model
  model.build_engine()

  # 2. regenerate with onnx from source model
  model.build_engine(input_shape="")

  # 3. regenerate with new onnx file
  model.build_engine(model_path='', input_shape='')

def test_train():
  model = loadNet(task="object_detection", model_type="yolov3", backbone="resnet18")
  model.train()

def test_preprocess():
  model = loadNet(task="object_detection", model_type="yolov3", backbone="resnet18")
  model.preprocess()


def test_general():
  # pytorch
  model = loadNet().troch()

  #onnx
  model = loadNet().onnx()

  #trt
  model = loadNet().build_engine()

def test_avaliablenet():
  main.avaliableNet()

def test_loadNet():
  #main.loadNet(task='classification',model_type='resnet18',backbone='resnet101')
  main.loadNet(task='object_detection',model_type='yolov3',backbone='resnet101')
  main.loadNet(task='object_detection',model_type='yolov4')

def test_onnx():
  model = main.loadNet(task='objedt_dwet',model_type='sdfo')

  
if __name__ == '__main__':
  #test_avaliablenet()
  test_loadNet()
