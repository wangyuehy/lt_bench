from common.util import dict_get,logging, printItem, check_parameter
from importlib import import_module
from datazoo import getDataDir
from omegaconf import OmegaConf
import pdb

modeldict={
  'classification':{
    'resnet18':"cv.classification.resnet18.resnet18",
    #'resnet101':'CV.classification.resnet101',
  },
  "object_detection":{
    'yolov3':{
      'resnet18':"cv.object_detection.yolov3.resnet18",
      #'resnet101':'CV.object_detection.yolov3.resnet101'
    }
  },
  "segmentation":{

  },
  "NLP":{

  }
}

def check_class_parameter(func):
  def wrapper(class_instance,*args, **kwargs):
    base_conf = class_instance.config
    config = base_conf.gent(func.__name__,[])
    for k in kwargs:
      if k not in config:
        raise TypeError
      config[k] = kwargs[k]

    #print(class_instance.__name__)
    return func(self,*args,config)
  return wrapper
 

def avaliableNet():
  '''
    return avaliable net list
  '''
  printItem(modeldict)

@check_parameter(base_yaml='base.yaml')
def loadNet(cfg):
  '''
    Function:
      return model base on parameter
      - task
        - model_type
          - backbone
  '''
  task = cfg.task
  model_type = cfg.model_type
  backbone = cfg.backbone
  if task in modeldict:
    modeltypes = modeldict[task]
    if model_type in modeltypes:
      backbones = modeltypes[model_type]
      if type(backbones) == str:
        if backbone:
          logging.warning('Unused parameter {}'.format(backbone))
        logging.info('import {}'.format(backbones))
        model = import_module(backbones).model
        return model
      elif type(backbones) == dict:
        if backbone in backbones:
          backbone_path = backbones[backbone]
          if type( backbone_path ) == str:
            logging.info('import {}'.format(backbone_path))
            model = import_module(backbone_path).model
            return model
          else:
            logging.error('not support model')
        else:
          logging.error('Invalid backbone {}'.format(backbone))
      else:
        logging.error('Invliad modeltype, check your modeldict {},{}'.format(task, model_type))
    else:
      logging.error('Invalid model_type: {}'.format(model_type))
  else:
    logging.error("Invalid task: {}".format(task))

class torchmodel:
  def __init__(self,model=None, model_path=None):
    self.model = model
    self.model_path = model_path
  
  def __call__(self, cfg):
    if cfg.model_path:
      return None
    elif self.model:
      return self
    else:
      return None 
class trtmodel:
  def __init__(self, model=None, model_path=None):
    self.model = model
    self.model_path = model_path

  def __call__(sel, **args):
    pass
class cvOnnxModel:
  def __init__(self,input_shape=[3,224,224], batch_size=-1, model_path=None,opset_version=11 ):
    self.input_shape = input_shape
    self.batch = batch_size
    self.model_path = model_path
    if self.model_path:
      self.model = onnx.load(self.model_path)
    else:
      self.model = None
    self.opset_version = opset_version

  def __call__(self, **args):
    input_shape = dict_get(args, 'input_shape', default=None)
    batch_size = dict_get(args, 'batch_size', default=None)
    opset_version = dict_get(args, 'opset_version', default=11)
    if shape != self.input_shape or batch != self.batch or opset_version != self.opset_version:
      return None
    return self

class baseModel(object):
  def __init__(self,args):
    self.torchmodel = None # key : shape, batch, model, model_path
    self.onnxmodel = None  #onnxmodel()  #{shape=3x320x320,  batch=1, model_path='///xd//x', model=onnx.load(model_path) }
    self.trtmodel = None # {shape=3x320x320,  batch=1, model_path='///xd//x', precesion=int8}
    self.batch_size = -1
    self.base_yaml = 'base.yaml'
    self.config = OmegaConf.load(self.base_yaml)  # used in wrapper check_wrapper_paramter

  def train(self,args):
    pass

  @check_class_parameter
  def infer(self,cfg):
    infer_type = cfg.infer_type
    if infer_type == 'torch':
      return self._torch_infer(kwargs)
    elif infer_type == 'onnx':
      return self._onnx_infer(kwargs)
    elif infer_type =='trt':
      return self._trt_infer(kwargs)
    else:
      logging.error('Invalid infer_type')

  @check_class_parameter
  def build_engine(self,cfg):
    engine_type = cfg.engine_type
    #force_generate=False,return_path=False, precison='fp16', batch_size=-1, engine_type='trt', ):
    #engine_type = dict_get(kwargs, "engine_type", default='trt')
    if engine_type =='trt':
      self._build_trt_engine(return_path=False, precison='fp16', batch_size=-1)
    else:
      logging.log('invalid trt type {}'.format(engine_type))

  def build_trt_engine(self,cfg):
    pass
  def preprocess(self,args):
    # parser whole folder
    pass

  def preprocess_one(self, cfg):
    # parse one batch, return numpy.array 
    pass

  def postprocess(self,args):
    pass

  @check_class_parameter
  def torch(self, cfg):
    if self.torchmodel(cfg):
      pass
    else:
      self.build_torch(cfg)
    if cfg.return_path:
      return self.trochmodel.model, self.torchmodel.model_path
    else:
      return self.torchmodel.model

  def onnx(self, cfg):
    if not self.onnxmodel(cfg):
      self._build_onnx(cfg)
    if cfg.return_path:
      return self.onnxmodel.model, self.onnxmodel.model_path
    else:
      return self.onnxmodel.model

  def trt(self, cfg):
    if not self.trtmodel(cfg):
      self.build_trt_engine(cfg)
    if cfg.return_path:
      return self.trtmodel.model, self.trtmodel.model_path
    else:
      return self.trtmodel.model

  def build_torch(self,cfg):
    pass

  def build_onnx(self,cfg):
    pass

  def trt_infer(self,cfg):
    pass

  def onnx_infer(self,cfg):
    pass

  def torch_infer(self,cfg):
    pass
