import torch
import torchvision.models
from common.util import depict_config,logging
from base import baseModel,check_class_parameter
import onnx
from cv.classification.trt_build import preprocess_classification
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
      model.load_state_dict(cfg.model_path)
    return model

  ############ onnx ################
  @check_class_parameter
  def build_onnx(self, cfg):
    if cfg.verbose:
      depict_config(cfg)
    pytorch_model = self.torch(model_path=cfg.torch_model_path)
    dummy_input = torch.randn([cfg.batch_size].extend(cfg.input_shape))
    torch.onnx.export(pytorch_model, dummy_input, cfg.onnx_model_path,verbose=cfg.verbose,opset_version=cfg.opset_version)
    return cfg.onnx_model_path

  ############ engine ################
  @check_class_parameter
  def build_trt_engine(self, cfg):
    if cfg.onnx_model_path:
      onnx_file_path = cfg.onnx_model_path
    else:
      _,onnx_file_path = self.onnx(input_shape=cfg.input_shape,batch_size=cfg.batch_size, return_path=True)
    if calibrator == 'simple':
      calibrator_func = classificationEntropyCalibrator

    trt_build.build_engine(onnx_file_path, engine_file_path=cfg.trt_model_path, batch_size=cfg.batch_size, \
      input_shape=cfg.input_shape, precision=cfg.precision, max_workspace_size=cfg.max_workspace_size, calib_cache=cfg.calib_cache,\
        calibrator=calibrator_func)


  def preprocess(self, cfg):
    input_shape = dict_get(kwargs, 'input_shape', default=(3,224,224))
    mean = dict_get(kwargs, 'mean', default=[0.485, 0.456, 0.406])
    std = dict_get(kwargs,'std', default=[0.229, 0.224, 0.225])
    build_dir = dict_get(kwargs,'build_dir', default=None)
    source_dir = dict_get(kwargs, 'source_dir', default=None)
    source_file = dict_get(kwargs, 'source_file', default=None)
    if not build_dir:
      build_dir = self.preprocess_data_dir
    if not cfg.build_dir:
      build_dir = cfg.preprocess_data_dir
    preprocess_classification(source_dir=cfg.source_file)
    preprocess_classification(source_dir=source_dir, source_file=source_file,\
      build_dir=build_dir, mean=mean, std=std,input_shape=input_shape)


 

