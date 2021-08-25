
import torchvision.models
from common.util import check_args, dict_get, depict_config
from base import baseModel,check_class_parameter
import onnx
from trt_build import preprocess_classification
from omegaconf import OmegaConf

Build_engine_params = []


class model(baseModel):
  
  def __init__(self):
    self.extend_yaml = 'cv/classification/resnet18/resnet18.yaml'
    self.config = OmegaConf.merge(OmegaConf.load(self.base_yaml), OmegaConf.load(self.extend_yaml))
  ############ torch ################
  
  @check_class_parameter
  def build_torch(self,cfg):
    # default model_path defined in yaml
    model = models.resnet18()
    if cfg.model_path:
      model.load_state_dict(torch.load(cfg.model_path))
    elif cfg.load_pretrained:
      model.load_state_dict(torch.load(cfg.pretrained.model_path))
    else:
      logging.warning("Doesn't load any weight for resnet18")
    self.torchmodel = torchModel(model, cfg.pretrained.model_path)

  ############ onnx ################
  @check_class_parameter
  def build_onnx(self, cfg):
    if cfg.verbose:
      depict_config(cfg)
    # if defined cfg.torch_model_path, load it, if not, load from modelzoo
    pytorch_model = self.torch(model_path=cfg.torch_model_path)
    dummy_input = torch.randn([cfg.batch_size].extend(cfg.input_shape))
    torch.onnx.export(pytorch_model, dummy_input, cfg.onnx_model_path,verbose=cfg.verbose)
    self.onnxmodel = cvOnnxModel(batch_size=cfg.batch_size, input_shape=input_shape)
    



  ############ engine ################
  @check_class_parameter
  def build_trt_engine(self, cfg):
    if cfg.onnx_model_path:
      onnx_file_path = cfg.onnx_model_path
    else:
      _,onnx_file_path = self.onnx(input_shape=cfg.input_shape,batch_size=cfg.batch_size, return_path=True)
    if calibrator == 'simple':
      calibrator_func = classificationEntropyCalibrator

    trt_build.build_engine(onnx_file_path, engine_file_path=cfg.engine_model_path, batch_size=cfg.batch_size, \
      input_shape=cfg.input_shape, precision=cfg.precision, max_workspace_size=cfg.max_workspace_size, calib_cache=cfg.calib_cache,\
        calibrator=calibrator_func)

 

  @check_parameter(self.config)
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


 

