
from cv.classification.classifcation import classification
from omegaconf import OmegaConf


Build_engine_params = []


class model(classification):
  def __init__(self):
    self.extend_yaml = 'cv/classification/resnet18/resnet18_torch/resnet18.yaml'
    super(model,self).__init__()
   
   