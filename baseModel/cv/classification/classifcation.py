import torch
import torchvision.models
from common.util import depict_config,logging,set_file_dir
from base import baseModel,check_class_parameter
import onnx
import os
from cv.classification import trt_build
from cv.classification.trt_build import preprocess_imagenet_data
from omegaconf import OmegaConf


Build_engine_params = []


class classification(baseModel):
  
  def __init__(self):
    super(classification,self).__init__()
   
  ############ torch ################
  @check_class_parameter
  def build_torch(self,cfg):
    model = torchvision.models.resnet18()
    if cfg.model_path:
      model.load_state_dict(torch.load(cfg.model_path))
    return model

  ############ onnx ################
  @check_class_parameter
  def build_onnx(self, cfg):
    if cfg.verbose:
      depict_config(cfg)
    set_file_dir(cfg.onnx_model_path)
    pytorch_model = self.torch(model_path=cfg.torch_model_path)
    if cfg.batch_size > 0:
      dummy_input = torch.randn([cfg.batch_size, cfg.input_shape[0], cfg.input_shape[1], cfg.input_shape[2]])
      torch.onnx.export(pytorch_model, dummy_input, cfg.onnx_model_path,verbose=cfg.verbose,opset_version=cfg.opset_version)
    else:
      dummy_input = torch.randn([1, cfg.input_shape[0], cfg.input_shape[1], cfg.input_shape[2]] )
      torch.onnx.export(
        pytorch_model, dummy_input, cfg.onnx_model_path,  
        opset_version=cfg.opset_version,verbose=cfg.verbose, 
        input_names=['input'], output_names= ['output'],
        dynamic_axes={'input':{0:'batch_size'},
                      'output':{0:'batch_size'}}
      )
    if cfg.verbose:
      logging.info('build onnx to {}'.format(cfg.onnx_model_path))
    return cfg.onnx_model_path

  ############ engine ################
  @check_class_parameter
  def build_trt_engine(self, cfg):
    if cfg.onnx_model_path:
      onnx_file_path = cfg.onnx_model_path
    else:
      _,onnx_file_path = self.onnx(input_shape=cfg.input_shape,batch_size=cfg.batch_size, return_path="build/cv/classification/tmp.onnx")


    # if force calibrate or the preprocess dir doesn't exists, do the preprocess
    if cfg.precision == 'int8' and cfg.force_calibrate or not os.path.exists(cfg.calibrate_dir):
      self.preprocess(dst_dir=cfg.calibrate_dir, source_dir=cfg.source_dir, force_preprocess=True)
  
    trt_build.build_engine(onnx_file_path, \
        calibrate_dir=cfg.calibrate_dir, \
        engine_file_path=cfg.trt_model_path, 
        batch_size=cfg.batch_size, \
        input_shape=cfg.input_shape, 
        precision=cfg.precision, \
        explicit_batch=cfg.explicit_batch, \
        max_workspace_size=cfg.max_workspace_size, \
        calib_cache=cfg.calibrate_cache,\
        calibrator=cfg.calibrator, \
        max_batch_size=cfg.max_batch_size)
    return cfg.trt_model_path

  @check_class_parameter
  def preprocess_one(self, cfg):
    return preprocess_imagenet_data(cfg.source_name, cfg.input_shape, cfg.mean, cfg.std)


  def postprocess(self,cfg):
    pass
 

