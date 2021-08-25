#!/usr/bin/env python3

# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import math

import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)




def preprocess_imagenet_for_resnet50(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    """Proprocess the raw images for inference."""

    def loader(file):
        """Resize and crop image to required dims and return as FP32 array."""

        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = (224, 224)
        image = resize_with_aspectratio(image, h, w)
        image = center_crop(image, h, w)
        image = np.asarray(image, dtype='float32')
        # Normalize image.
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image -= means
        # Transpose.
        image = image.transpose([2, 0, 1])
        return image

    def quantizer(image):
        """Return quantized INT8 image of input FP32 image."""
        return np.clip(image, -128.0, 127.0).astype(dtype=np.int8, order='C')

    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
                         "data_maps/imagenet/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
                         "data_maps/imagenet/val_map.txt", formats, overwrite)


class BenchmarkBuilder(AbstractBuilder):
    def __init__(self, args, name="", workspace_size=(1 << 30), modelname=""):
      self.name = name
      self.args = args
      self.modelname = modelname
      # Configuration variables
      self.verbose = dict_get(args, "verbose", default=False)
      if self.verbose:
        logging.info("========= BenchmarkBuilder Arguments =========")
        for arg in args:
          logging.info("{:}={:}".format(arg, args[arg]))

      self.system_id = args["system_id"]
      self.scenario = args["scenario"]
      self.config_ver = args["config_ver"]
      self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario)

      # Set up logger, builder, and network.
      self.logger = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO)
      trt.init_libnvinfer_plugins(self.logger, "")
      self.builder = trt.Builder(self.logger)
      self.builder_config = self.builder.create_builder_config()
      self.builder_config.max_workspace_size = workspace_size
      if dict_get(args, "verbose_nvtx", default=False):
        self.builder_config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

      # Precision variables
      self.input_dtype = dict_get(args, "input_dtype", default="fp32")
      self.input_format = dict_get(args, "input_format", default="linear")
      self.precision = dict_get(args, "precision", default="int8")
      self.clear_flag(trt.BuilderFlag.TF32)
      if self.precision == "fp16":
        self.apply_flag(trt.BuilderFlag.FP16)
      elif self.precision == "int8":
        self.apply_flag(trt.BuilderFlag.INT8)
      self.batch_size = 1
        

      # Currently, TRT has limitation that we can only create one execution
      # context for each optimization profile. Therefore, create more profiles
      # so that LWIS can create multiple contexts.
      self.num_profiles = self.args.get("gpu_copy_streams", 4)
      self.initialized = False

    def apply_flag(self, flag):
        """Apply a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) | (1 << int(flag))

    def clear_flag(self, flag):
        """Clear a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) & ~(1 << int(flag))

    def _get_engine_fpath(self, device_type, batch_size):
        # Use default if not set
        if device_type is None:
            device_type = self.device_type
        if batch_size is None:
            batch_size = self.batch_size

        # If the name ends with .plan, we assume that it is a custom path / filename
        if self.name.endswith(".plan"):
            return "{:}/{:}".format(self.engine_dir, self.name)
        else:
            if self.modelname:
                return "{:}/{:}-{:}-{:}-b{:}-{:}.{:}.plan".format(
                self.engine_dir, self.modelname, self.scenario,
                device_type, batch_size, self.precision, self.config_ver)
            return "{:}/{:}-{:}-{:}-b{:}-{:}.{:}.plan".format(
                self.engine_dir, self.name, self.scenario,
                device_type, batch_size, self.precision, self.config_ver)

    def build_engines(self):
        """Calls self.initialize() if it has not been called yet. Builds and saves the engine."""

        if not self.initialized:
            self.initialize()

        # Create output directory if it does not exist.
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        engine_name = self._get_engine_fpath(self.device_type, self.batch_size)
        logging.info("Building {:}".format(engine_name))

        if self.network.has_implicit_batch_dimension:
            self.builder.max_batch_size = self.batch_size
        else:
            self.profiles = []
            # Create optimization profiles if on GPU
            if self.dla_core is None:
                for i in range(self.num_profiles):
                    profile = self.builder.create_optimization_profile()
                    for input_idx in range(self.network.num_inputs):
                        input_shape = self.network.get_input(input_idx).shape
                        input_name = self.network.get_input(input_idx).name
                        min_shape = trt.Dims(input_shape)
                        min_shape[0] = 1
                        max_shape = trt.Dims(input_shape)
                        max_shape[0] = self.batch_size
                        profile.set_shape(input_name, min_shape, max_shape, max_shape)
                    if not profile:
                        raise RuntimeError("Invalid optimization profile!")
                    self.builder_config.add_optimization_profile(profile)
                    self.profiles.append(profile)
            else:
                # Use fixed batch size if on DLA
                for input_idx in range(self.network.num_inputs):
                    input_shape = self.network.get_input(input_idx).shape
                    input_shape[0] = self.batch_size
                    self.network.get_input(input_idx).shape = input_shape

        # Build engines
        engine = self.builder.build_engine(self.network, self.builder_config)
        buf = engine.serialize()
        with open(engine_name, 'wb') as f:
            f.write(buf)

    def calibrate(self):
        """Generate a new calibration cache."""

        self.need_calibration = True
        self.calibrator.clear_cache()
        self.initialize()
        # Generate a dummy engine to generate a new calibration cache.
        if self.network.has_implicit_batch_dimension:
            self.builder.max_batch_size = 1
        else:
            for input_idx in range(self.network.num_inputs):
                input_shape = self.network.get_input(input_idx).shape
                input_shape[0] = 1
                self.network.get_input(input_idx).shape = input_shape
        engine = self.builder.build_engine(self.network, self.builder_config)



asdfdfdfsdfsdfsdgsdfsdgsdfsdfsdfdfssdsddfsdfssdwhois that wher i can not found the real
dangerouse way of w 

def add_profiles(config, inputs, opt_profiles):
    logger.debug("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)


def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        logger.error("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))


def check_network(network):
    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    logger.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logger.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))


def get_batch_sizes(max_batch_size):
    # Returns powers of 2, up to and including max_batch_size
    max_exponent = math.log2(max_batch_size)
    for i in range(int(max_exponent)+1):
        batch_size = 2**i
        yield batch_size
    
    if max_batch_size != batch_size:
        yield max_batch_size


# TODO: This only covers dynamic shape for batch size, not dynamic shape for other dimensions
def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            print("fbs:{}".format(fbs))
            print("shape:{}".format(shape))
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]

            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))

    return list(profiles.values())

def onnx_to_tensorrt(onnx,
    output='model.engine',
    max_batch_size=32,
    verbosity=None,
    explicit_batch=False,
    explicit_precision=False,
    gpu_fallback=False,
    refittable=False,
    debug=False,
    strict_types=False,
    fp16=False,
    int8=False,
    calibration_cache="calibration.cache",
    calibration_data=None,
    calibration_batch_size=32,
    max_calibration_size=512,
    preprocess_func=None,
    simple=False):
    # Adjust logging verbosity
    if verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
    logger.info("TRT_LOGGER Verbosity: {:}".format(TRT_LOGGER.min_severity))

    # Network flags
    network_flags = 0
    if explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if explicit_precision:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
    }

    # Building engine
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(network_flags) as network, \
         builder.create_builder_config() as config, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
            
        config.max_workspace_size = 2**30 # 1GiB

        # Set Builder Config Flags
        for flag in builder_flag_map:
            if eval(flag):
                logger.info("Setting {}".format(builder_flag_map[flag]))
                config.set_flag(builder_flag_map[flag])

        # Fill network atrributes with information by parsing model
        with open(onnx, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(onnx))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        # Display network info and check certain properties
        check_network(network)

        if explicit_batch:
            print("if args.explicit_batch")
            # Add optimization profiles
            batch_sizes = [1, 8, 16, 32, 64]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
            #builder.max_batch_size = max_batch_size
            add_profiles(config, inputs, opt_profiles)
        # Implicit Batch Network
        else:
            print("else args.explicit_batch")
            builder.max_batch_size = max_batch_size
            opt_profiles = []

        # Precision flags
        if fp16 and not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform.")

        if int8 and not builder.platform_has_fast_int8:
            logger.warning("INT8 not supported on this platform.")

        if int8:
            if simple:
                from .SimpleCalibrator import SimpleCalibrator # local module
                config.int8_calibrator = SimpleCalibrator(network, config)
            else:
                from .ImagenetCalibrator import ImagenetCalibrator, get_int8_calibrator # local module
                config.int8_calibrator = get_int8_calibrator(calibration_cache,
                                                             calibration_data,
                                                             max_calibration_size,
                                                             preprocess_func,
                                                             calibration_batch_size)
        log_info = output.rstrip('/')+'.log'
        if int8:
            cmd = 'trtexec --onnx={} --int8 --saveEngine={}  --explicitBatch > {}'.format(onnx,output,log_info)
        elif fp16:
            cmd = 'trtexec --onnx={} --fp16 --saveEngine={} --explicitBatch > {}'.format(onnx,output,log_info)
        logger.info('exectue cmd'+cmd)
        print(cmd)
        # Attention !!!
        # using this model is extremly unstable for workstation, it will restart many times. 
        # so generate the bash script and execute it yourself
        #os.system(cmd)

        # replace build_engine with trtexec, 20210608
        # add config file to generate is more stable, 20210611
        logger.info("Building Engine...")
        with builder.build_engine(network, config) as engine, open(output, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(output))
            f.write(engine.serialize())
    



from __future__ import print_function

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import pdb
import numpy as np
import sys, os
TRT_LOGGER = trt.Logger()
#TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
from itertools import chain
import argparse
import os


try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


######### prepare calibrate data #############

def cv2_preprocess(img,target_size=(416,416),mean=[0,0,0],std=[255.,255.,255.]):
  h,w,c = img.shape
  #target_size = 416
  #mean = [0,0,0]
  #std=[255.,255.,255.]
  mean = np.float64(np.array(mean).reshape(1,-1))
  to_rgb=True
  stdinv = 1/np.float64(np.array(std).reshape(1,-1))

  # resize
  scale_h = target_size[0]/h
  scale_w = target_size[1]/w
  if target_size[0]/ h > target_size[1]/w:
    scale = target_size[1]/w
  else:
    scale = target_size[0]/h
  dim = (int(w*scale),int(h*scale))  # notice w,hw sequence
  res_img = cv2.resize(img, dim,interpolation=cv2.INTER_LINEAR)
  resized_img = res_img.copy().astype(np.float32)

  # norm
  if to_rgb:
    cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB, resized_img)
  cv2.subtract(resized_img, mean, resized_img)
  cv2.multiply(resized_img, stdinv, resized_img)

  # pad
  padding = (0,0, target_size[1] - resized_img.shape[1], target_size[0] - resized_img.shape[0])
  pad_img = cv2.copyMakeBorder(resized_img, padding[1],padding[3],padding[0],padding[2],
                        cv2.BORDER_CONSTANT,value=0)

  # transpose
  img = np.ascontiguousarray(pad_img.transpose(2,0,1))

  return img
  
def imgs_preprocess(imgs,target_size=(416,416)):
  #target_size = 416
  mean = [0,0,0]
  std=[255.,255.,255.]

  datas = np.random.random((len(imgs),3,target_size[0],target_size[1]))
  for ith,img in enumerate(imgs):
    fimg = cv2.imread(img)
    preprocessed_img = cv2_preprocess(fimg,target_size=target_size,mean=mean,std=std)
    datas[ith,:]=preprocessed_img
  return datas

def load_COCO_data(num_cali=100, target_size=(416,416)):
    coco_path = '/workspace/public/dataset/cv/coco/val2017'
    imgindex = np.random.randint(0,5000,num_cali)
    #imglist = os.listdir(coco_path)[:num_cali]
    imglist = os.listdir(coco_path)
    imglist = [os.path.join(coco_path,imglist[inx]) for inx in imgindex]
    datas = imgs_preprocess(imglist,target_size)
    return datas


class CoCoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=8, input_shape=(416,416)):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
  
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_COCO_data(8, input_shape)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            print('read cache ',self.cache_file)
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        print('write cache ',cache)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        print("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))

def check_network(network):
    if not network.num_outputs:
        print("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    print("=== Network Description ===")
    for i, inp in enumerate(inputs):
        print("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        print("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))

def add_profiles(config, inputs, opt_profiles):

    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
        config.add_optimization_profile(profile)

# TODO: This only covers dynamic shape for batch size, not dynamic shape for other dimensions
def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            print("fbs:{}".format(fbs))
            print("shape:{}".format(shape))
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]
            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))
    return list(profiles.values())



def gen_engine(onnx_file_path, engine_file_path="", batch_size=8,HW=(416,416),datatype=1):  #datatype 2 for fp16, 1 for int8, 0 for fp32
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder,\
            builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                    trt.OnnxParser(network, TRT_LOGGER) as parser:
            print('building batchsize {} input shape {}'.format(batch_size,HW))
            #batch_size = global_batch_size
            builder.max_batch_size = batch_size
            #config.max_workspace_size = 1 << 33 # 8GB
            #max_workspace_size = 1 << 30
            max_workspace_size = 1 <<31
            config.max_workspace_size = max_workspace_size
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            check_network(network)    # not neccessory

            # int8
            if datatype == 1:   
                config.set_flag(trt.BuilderFlag.INT8)
                calibration_cache = "coco_calibration.cache"
                #calibration_cache="/workspace/coco_calibration.cache"
                #if os.path.exists(calibration_cache):
                #    os.remove(calibration_cache)
                #    print('remove ',calibration_cache)
                calib = CoCoEntropyCalibrator(cache_file=calibration_cache, batch_size=batch_size,input_shape=HW)
                config.int8_calibrator = calib
            elif datatype ==2:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                pass
            #explicit_batch = True
            #if explicit_batch:
            batch_sizes = [batch_size]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
            add_profiles(config, inputs, opt_profiles)
            #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()






def main():
    parser = argparse.ArgumentParser(description="Creates a TensorRT engine from the provided ONNX file.\n")
    parser.add_argument("--onnx", required=True, help="The ONNX model file to convert to TensorRT")
    parser.add_argument("-o", "--output", type=str, default="model.engine", help="The path at which to write the engine")
    parser.add_argument("-b", "--max-batch-size", type=int, default=32, help="The max batch size for the TensorRT engine input")
    parser.add_argument("-v", "--verbosity", action="count", help="Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.")
    parser.add_argument("--explicit-batch", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.")
    parser.add_argument("--explicit-precision", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION.")
    parser.add_argument("--gpu-fallback", action='store_true', help="Set trt.BuilderFlag.GPU_FALLBACK.")
    parser.add_argument("--refittable", action='store_true', help="Set trt.BuilderFlag.REFIT.")
    parser.add_argument("--debug", action='store_true', help="Set trt.BuilderFlag.DEBUG.")
    parser.add_argument("--strict-types", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    parser.add_argument("--fp16", action="store_true", help="Attempt to use FP16 kernels when possible.")
    parser.add_argument("--int8", action="store_true", help="Attempt to use INT8 kernels when possible. This should generally be used in addition to the --fp16 flag. \
                                                             ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.")
    parser.add_argument("--calibration-cache", help="(INT8 ONLY) The path to read/write from calibration cache.", default="calibration.cache")
    parser.add_argument("--calibration-data", help="(INT8 ONLY) The directory containing {*.jpg, *.jpeg, *.png} files to use for calibration. (ex: Imagenet Validation Set)", default=None)
    parser.add_argument("--calibration-batch-size", help="(INT8 ONLY) The batch size to use during calibration.", type=int, default=32)
    parser.add_argument("--max-calibration-size", help="(INT8 ONLY) The max number of data to calibrate on from --calibration-data.", type=int, default=512)
    parser.add_argument("-p", "--preprocess_func", type=str, default=None, help="(INT8 ONLY) Function defined in 'processing.py' to use for pre-processing calibration data.")
    parser.add_argument("-s", "--simple", action="store_true", help="Use SimpleCalibrator with random data instead of ImagenetCalibrator for INT8 calibration.")
    args, _ = parser.parse_known_args()

    # Adjust logging verbosity
    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif args.verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
    logger.info("TRT_LOGGER Verbosity: {:}".format(TRT_LOGGER.min_severity))

    # Network flags
    network_flags = 0
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.explicit_precision:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
    }

    # Building engine
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(network_flags) as network, \
         builder.create_builder_config() as config, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
            
        config.max_workspace_size = 2**30 # 1GiB

        # Set Builder Config Flags
        for flag in builder_flag_map:
            if getattr(args, flag):
                logger.info("Setting {}".format(builder_flag_map[flag]))
                config.set_flag(builder_flag_map[flag])

        # Fill network atrributes with information by parsing model
        with open(args.onnx, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(args.onnx))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        # Display network info and check certain properties
        check_network(network)

        if args.explicit_batch:
            print("if args.explicit_batch")
            # Add optimization profiles
            batch_sizes = [1, 8, 16, 32, 64]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
            add_profiles(config, inputs, opt_profiles)
        # Implicit Batch Network
        else:
            print("else args.explicit_batch")
            builder.max_batch_size = args.max_batch_size
            opt_profiles = []

        # Precision flags
        if args.fp16 and not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform.")

        if args.int8 and not builder.platform_has_fast_int8:
            logger.warning("INT8 not supported on this platform.")

        if args.int8:
            if args.simple:
                from SimpleCalibrator import SimpleCalibrator # local module
                config.int8_calibrator = SimpleCalibrator(network, config)
            else:
                from ImagenetCalibrator import ImagenetCalibrator, get_int8_calibrator # local module
                config.int8_calibrator = get_int8_calibrator(args.calibration_cache,
                                                             args.calibration_data,
                                                             args.max_calibration_size,
                                                             args.preprocess_func,
                                                             args.calibration_batch_size)

        logger.info("Building Engine...")
        with builder.build_engine(network, config) as engine, open(args.output, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(args.output))
            f.write(engine.serialize())

if __name__ == "__main__":
    main()