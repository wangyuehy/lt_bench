from common.util import dict_get,logging, printItem, check_parameter,get_absolute_config, set_dir,set_file_dir
from importlib import import_module
import os
from datazoo import getDataDir
from omegaconf import OmegaConf, DictConfig
from register import modeldict
import copy
import pdb
import tensorrt as trt
import torch
import onnx
import numpy as np
from abc import ABC, abstractmethod, abstractstaticmethod
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def check_class_parameter(func):
  def wrapper(class_instance, *args, **kwargs):
    base_conf = copy.deepcopy(class_instance.config)
    config = get_absolute_config(base_conf, func.__name__)
    for k in kwargs:
      if k not in config:
        raise TypeError("{} not found".format(k))
      config[k] = kwargs[k]
    if len(args) == 1 and type(args[0]) == DictConfig:
      config = OmegaConf.merge(config,args[0])
      return func(class_instance,config)
    return func(class_instance,*args,config)
  return wrapper
 
def avaliableNet():
  '''
    return avaliable net list
  '''
  printItem(modeldict)

@check_parameter(base_yaml=os.path.join(os.path.dirname(__file__), 'base.yaml'))
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

class baseModel(ABC):
  def __init__(self):
    self.batch_size = -1
    self.base_yaml = os.path.join(os.path.dirname(__file__), 'base.yaml')
    try:
      if self.extend_yaml:
        self.config = OmegaConf.merge(OmegaConf.load(self.base_yaml), OmegaConf.load(self.extend_yaml))
      else:
        self.config = OmegaConf.load(self.base_yaml)  
    except AttributeError as e:
      logging.error('extend_yaml not defined')
      raise AttributeError('extend_yaml not defined')

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
    if engine_type =='trt':
      self._build_trt_engine(return_path=False, precison='fp16', batch_size=-1)
    else:
      logging.log('invalid trt type {}'.format(engine_type))

  @abstractmethod
  def build_trt_engine(self,cfg):
    pass

  @check_class_parameter
  def preprocess(self,cfg):
    if os.path.exists(cfg.dst_dir) and not cfg.force_preprocess:
      logging.info(f'dst_dir={cfg.dst_dir} already exist')
      return
    else:
      if not os.path.exists(cfg.source_dir):
        logging.error('source dir does not exists {}'.format(cfg.source_dir))
      filelist = []
      if cfg.source_file:
        if not os.path.exists(cfg.source_file):
          logging.error('source file does not exists {}'.format(cfg.source_file))
        with open(cfg.source_file, 'r') as f:
          for line in f.readlines():
            relpath = line.strip().split()[0]
            if relpath:
              filelist.append(relpath)
      else:
        # find files in source_dir recursively
        for root, _, files in os.walk(cfg.source_dir):
          startidx = root.find(cfg.source_dir) + len(cfg.source_dir) + 1
          subdir = root[startidx:]
          filelist += [os.path.join(subdir, file) for file in files]
      # create dst_dir recursively
      dst_dir_list = set([os.path.join(cfg.dst_dir, os.path.dirname(p)) for p in filelist])
      for d in dst_dir_list:
        set_dir(d)
      logging.info(f'start to process {len(filelist)} images')
      for relpath in filelist:
        srcname = os.path.join(cfg.source_dir, relpath)
        preprocessed_name = os.path.join(cfg.dst_dir, relpath)+'.npy'
        cfg.source_name = srcname
        preprocessed_data = self.preprocess_one(cfg)
        np.save(preprocessed_name, preprocessed_data)
        if cfg.verbose:
          logging.info('build data from {} to {}'.format(srcname, preprocessed_name))
      return 

  @abstractmethod
  def preprocess_one(self, cfg):
    pass
  @abstractmethod
  def postprocess(self,args):
    pass

  @check_class_parameter
  def torch(self, cfg):
    if not cfg.model_path and cfg.load_pretrained:
      cfg.model_path = cfg.pretrained.model_path
    model = self.build_torch(cfg,model_path=cfg.model_path)

    if cfg.return_path:
      set_file_dir(cfg.return_path)
      torch.save(model.state_dict(), cfg.return_path)
      return model, cfg.return_path
    else:
      return model
  
  @check_class_parameter
  def onnx(self, cfg):
    # if given onnx path, use the onnx model path
    if not cfg.model_path:
      raise ValueError('model path must be defined')
    if os.path.isfile(cfg.model_path):
      model_path = cfg.model_path
    # if load_pretrained and input parameters are same, use the pretrained onnx
    elif cfg.load_pretrained and self.Is_Pretrained_Onnx_Matched(cfg):
      model_path = cfg.pretrained.model_path
    else:
      # otherwise, build onnx model
      cfg['onnx_model_path'] = cfg.model_path 
      model_path = self.build_onnx(cfg)
    model = onnx.load(model_path)
    if cfg.return_path:
      return model, model_path
    else:
      return model

  @check_class_parameter
  def Is_Pretrained_Onnx_Matched(self, cfg):
    if cfg.input_shape != cfg.pretrained.input_shape or \
      cfg.batch_size != cfg.pretrained.batch_size or \
        cfg.opset_version != cfg.pretrained.opset_version:
      return False
    else:
      return True

  @check_class_parameter
  def trt(self, cfg):
    if  cfg.model_path:
      pass
    elif cfg.load_pretrained and self.Is_Pretrained_Trt_Matched(cfg):
      cfg.model_path = cfg['pretrained'][cfg.precision]['model_path']
    else:
      if cfg.return_path:
        cfg.trt_model_path = cfg.return_path
      cfg.model_path = self.build_trt_engine(cfg)
    
    with open(cfg.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
      model = runtime.deserialize_cuda_engine(f.read())

    if cfg.return_path:
      if cfg.return_path != cfg.model_path:
        set_file_dir(cfg.return_path)
        os.system('cp {} {}'.format(cfg.model_path, cfg.return_path))
      return model,cfg.model_path
    else:
      return model 

  @check_class_parameter
  def Is_Pretrained_Trt_Matched(self,cfg):
    pretrained_cfg = cfg.pretrained[cfg.precision]
    if cfg.input_shape != pretrained_cfg.input_shape or \
      cfg.batch_size != pretrained_cfg.batch_size or \
        cfg.precision != pretrained_cfg.precision:
      return False
    else:
      return True
  @abstractmethod
  def build_torch(self,cfg):
    pass
  @abstractmethod
  def build_onnx(self,cfg):
    pass
  #@abstractmethod
  def trt_infer(self,cfg):
    pass
  #@abstractmethod
  def onnx_infer(self,cfg):
    pass
  #@abstractmethod
  def torch_infer(self,cfg):
    pass
