import os
from cv.object_detection.ssd.ssd import SSD
from omegaconf import OmegaConf
from base import baseModel,check_class_parameter

class model(SSD):
    def __init__(self):
        self.backbone = 'mobilenetv2'
        self.extend_yaml = 'cv/object_detection/ssd/ssd_mobilenetv2_torch/ssd_mobilenetv2.yaml'
        super().__init__()

    @check_class_parameter
    def build_torch(self,cfg):
        return super().build_model(config_path=cfg.model_cfg, checkpoint_path=cfg.model_path)

    @check_class_parameter
    def build_onnx(self, cfg):
        return super().build_onnx(cfg)

    @check_class_parameter
    def build_trt_engine(self, cfg):
        return super().build_trt_engine(cfg)

    def preprocess(self, cfg):
        super().preprocess(cfg)

    def preprocess_one(self, cfg):
        super().preprocess_one(cfg)

    def postprocess(self, cfg):
        pass

    def get_normalize_cfg(self):
        return [123.675, 116.28, 103.53], [1, 1, 1]



        