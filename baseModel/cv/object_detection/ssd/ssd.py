
import sys
import os
from functools import partial
from abc import abstractmethod
import copy

import numpy as np
import torch

from base import baseModel
from common.util import depict_config,logging

sys.path.append('/home/jliu/codes/model_convert') # TODO
from onnx_to_tensorrt_api import onnx_to_tensorrt
from cv.object_detection.ssd import onnx_modifier

class SSD(baseModel):
    def __init__(self):
        super().__init__()

    def build_onnx(self, cfg):
        shape = cfg.input_shape
        batchsize = cfg.batch_size
        opset_version = cfg.opset_version
        save_path_ = cfg.onnx_model_path
        verbose = cfg.verbose
        if verbose:
            depict_config(cfg)

        # get torch model
        torch_model = self.torch(model_path=cfg.torch_model_path)

        # create dummy input, see mmdetection/mmdet/core/export/pytorch2onnx.py:generate_inputs_and_wrap_model
        one_img = torch.rand(3, *shape).requires_grad_() # torch.float32
        C, H, W = shape[0], shape[1], shape[2]
        one_meta = {
            'img_shape': (H, W, C),
            'ori_shape': (H, W, C),
            'pad_shape': (H, W, C),
            'filename': None,
            'scale_factor': 1.0,
            'flip': False,
            'show_img': None,
        }
        batch_input = [one_img] * (batchsize if batchsize > 0 else 1)

        # prepare model for export
        img_metas = [ [one_meta] * (batchsize if batchsize > 0 else 1) ]
        torch_model.forward = partial(torch_model.forward, img_metas=img_metas, return_loss=False)
        try:
            from mmcv.onnx.symbolic import register_extra_symbolics
        except ModuleNotFoundError:
            raise NotImplementedError('please update mmcv to version>=v1.0.4')
        register_extra_symbolics(opset_version)

        # export
        dynamic_axes = None if batchsize > 0 else {'input':{0}, 'dets':{0}, 'labels':{0}}
        torch.onnx.export(
                torch_model,
                batch_input,
                save_path_,
                input_names=['input'],
                output_names=['dets', 'labels'],
                export_params=True,
                keep_initializers_as_inputs=True,
                do_constant_folding=True,
                verbose=verbose,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes)
        return save_path_

    def build_trt_engine(self, cfg):
        input_shape = cfg.input_shape
        batch_size = cfg.batch_size
        precision = cfg.precision
        out_engine_path = cfg.trt_model_path
        calibration_style = cfg.calibration_style
        calibration_data = cfg.calibration_data
        max_calibration_size = cfg.get('max_calibration_size', 500)
        calibration_batch_size = cfg.get('calibration_batch_size', 32)
        calibration_cache_path = cfg.calibration_cache_path
        verbosity = cfg.get('verbosity', 'verbose')
        
        # get onnx
        if cfg.onnx_model_path:
            onnx_file_path = cfg.onnx_model_path
        else:
            _,onnx_file_path = self.onnx(input_shape=cfg.input_shape,batch_size=cfg.batch_size, return_path=True) # TODO
        
        # adapt onnx to trt engine
        modified_onnx_path = 'tmp.onnx'
        modified_onnx_path = self._adapt_onnx_to_trt(onnx_file_path, modified_onnx_path)

        if precision == 'int8':
            if calibration_style == 'simple':
                print('use simple calibration')
                onnx_to_tensorrt(modified_onnx_path, 
                        output=out_engine_path, 
                        int8=True, 
                        fp16=False, 
                        simple=True,
                        explicit_batch=True, 
                        cali_input_shape=tuple(input_shape),
                        verbosity=None if verbosity=="err" else (1 if verbosity =="info" else 2))
            else:
                if self.backbone in ['vgg', 'mobilenetv2']:
                    preprocess_func='preprocess_coco_mmdet_ssd'
                onnx_to_tensorrt(modified_onnx_path, 
                        output=out_engine_path, 
                        int8=True, 
                        fp16=False, 
                        max_calibration_size=max_calibration_size, 
                        calibration_batch_size=calibration_batch_size, 
                        calibration_cache=calibration_cache_path, 
                        calibration_data=calibration_data, 
                        preprocess_func=preprocess_func, 
                        explicit_batch=True, 
                        use_cache_if_exists=False,
                        save_cache_if_exists=True,
                        cali_input_shape=tuple(input_shape),
                        verbosity=None if verbosity=="err" else (1 if verbosity =="info" else 2))
              
        else:
            onnx_to_tensorrt(modified_onnx_path, 
                        output=out_engine_path, 
                        int8=False, 
                        fp16=(precision=='fp16'), 
                        explicit_batch=True, 
                        cali_input_shape=tuple(input_shape),
                        verbosity=None if verbosity=="err" else (1 if verbosity =="info" else 2))

        return out_engine_path
        

    def _adapt_onnx_to_trt(self, input_model_or_path=None, out_model_path=None):
        '''
        onnx_model_or_path: model or str, if None, use self.onnx_model
        out_model_path: str or None, if None, output the model directly and not save to file
        output:
            out_model_or_path: model or str, according to out_model_path
        '''
        input_model_or_path_ = input_model_or_path if input_model_or_path is not None else self.onnx_model
        out_model_or_path = onnx_modifier.modify_onnx(input_model_or_path_, out_model_path=out_model_path)
        return out_model_or_path

    # def preprocess(self, cfg):
    #     force_preprocess = cfg.get('force_preprocess', True)
    #     input_shape = cfg.input_shape
    #     mean_internel, std_internel = self.normalize_cfg
    #     mean = cfg.get('mean', mean_internel)
    #     std = cfg.get('std', std_internel)
    #     source_dir = cfg.source_dir
    #     source_file = cfg.source_file
    #     build_dir = cfg.build_dir
    #     if os.path.exists(build_dir) and not force_preprocess:
    #         return
    #     if not os.path.exists(build_dir):
    #         os.makedirs(build_dir)
    #     if not os.path.exists(source_dir):
    #         logging.error('source dir not existed %s'%source_dir)
    #     # get image file list
    #     imglist = []
    #     if os.path.exists(source_file):
    #         with open(source_file,'r') as f:
    #             for line in f.readlines():
    #                 imgname = line.strip()
    #                 imglist.append(os.path.join(source_dir, imgname))
    #     else:
    #         for root, _, files in os.walk(source_dir):
    #             imglist += [os.path.join(root, file) for file in files]
    #     # read, process and save
    #     ori_img = cfg.get('img', None)
    #     for img in imglist:
    #         cfg.img = img
    #         preprocessed_data = self.preprocess_one(cfg)
    #         preprocessed_name = os.path.join(build_dir, img)
    #         np.save(preprocessed_name, preprocessed_data)
    #     if ori_img:
    #         cfg.img = ori_img

    def preprocess_one(self, cfg):
        """Pre-processing a image with resize and normalization, non inplace modification
        img: imagepath or PIL.Image or 2D/3D np.array. If backend='PIL', input order is HWC-RGB; if backend='cv2', input order is HWC-BGR
        backend: 'PIL' or 'cv2'
        channels: int, desired number of channels, usually 1 or 3
        input_shape: shape of object image, (channel, height, width)
        Returns:
            img_data: 3D np.array of fp32, the order is CHW-BGR
        """
        img = cfg.source_name
        backend = cfg.get('backend', 'cv2')
        channels = cfg.get('channels', 3)
        input_shape = cfg.get('input_shape', [3, 224, 224])
        mean = cfg.get('mean', [123.675, 116.28, 103.53])
        std = cfg.get('std', [58.395, 57.12, 57.375])
        height, width = input_shape[1], input_shape[2]
        data = copy.deepcopy(img)
        if backend == 'PIL':
            from PIL import Image
            if isinstance(data, str):
                data = Image.open(data)
            elif isinstance(data, np.ndarray):
                data = Image.fromarray(data)
            data = data.resize((width, height), Image.ANTIALIAS)
            data = np.asarray(data).astype(np.float32)
            if len(data.shape) == 2:
                data = np.stack([data] * channels) # add dimension of channel
                print(f'accept grayscale image, preprocess to {channels} channels')
            else:
                data = data.transpose([2, 0, 1]) # to CHW
            mean_vec = np.array(mean)
            stddev_vec = np.array(std)
            assert data.shape[0] == channels
            for i in range(data.shape[0]):
                data[i, :, :] = (data[i, :, :] - mean_vec[i]) / stddev_vec[i]
        elif backend == 'cv2':
            import cv2
            if isinstance(data, str):
                data = cv2.imread(data)
            elif not isinstance(data, np.ndarray):
                raise TypeError('input must be image path or np.array for backend cv2')
            if len(data.shape) == 2:
                data = np.stack([data] * channels, axis=-1) # add dimension of channel
                print(f'accept grayscale image, preprocess to {channels} channels')
            data = cv2.resize(data, (width, height))
            data = data.astype(np.float32)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            mean = np.array([mean], dtype=np.float64)
            inv_std = 1 / np.array([std], dtype=np.float64)
            cv2.subtract(data, mean, data)
            cv2.multiply(data, inv_std, data)
            data = data.transpose(2, 0, 1) # to CHW
        return data

    @abstractmethod
    def get_normalize_cfg(self):
        pass

    @staticmethod
    def build_model(config_path=None, checkpoint_path=None):
        '''
        (re)build torch model with specified checkpoint and config(optional)
        config_path: str
        checkpoint: filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        output:
            torch model
        '''
        # modified from mmdetection/mmdet/core/export/pytorch2onnx.py:build_model_from_cfg
        model = None
        if config_path:
            from mmdet.models import build_detector
            import mmcv
            cfg = mmcv.Config.fromfile(config_path)
            import pickle
            f = open('/home/jliu/data/debug/loadcfg1_my.pkl', 'wb')
            pickle.dump(cfg.model, f)
            # import modules from string list.
            if cfg.get('custom_imports', None):
                from mmcv.utils import import_modules_from_strings
                import_modules_from_strings(**cfg['custom_imports'])
            # set cudnn_benchmark
            if cfg.get('cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = True
            cfg.model.pretrained = None
            cfg.data.test.test_mode = True
            # build the model
            cfg.model.train_cfg = None
            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        import pickle
        f = open('/home/jliu/data/debug/modelcfg1_my.pkl', 'wb')
        pickle.dump(cfg.model, f)
        f = open('/home/jliu/data/debug/model1_my.pkl', 'wb')
        pickle.dump(model, f)
        # load checkpoint
        if checkpoint_path:
            from mmcv.runner import load_checkpoint
            load_checkpoint(model, checkpoint_path, map_location='cpu')
        if model:
            model.cpu().eval()
        return model