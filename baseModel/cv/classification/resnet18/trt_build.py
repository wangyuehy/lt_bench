from common.util import logging
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit


########### preprocess the input data  ###########

def resize_with_aspectratio(img, out_height, out_width, scale=87.5):
    """Use OpenCV to resize image."""
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

def preprocess_imagenet_data(imgname, input_shape, mean, std):
  target_h, target_w = input_shape[1], input_shape[2]
  image = cv2.imread(imgname)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = resize_with_aspectratio(image, target_h, target_w)
  #image = center_crop(image, target_h, target_w)
  image = np.asarray(image, dtype='float32')

  means = np.array(mean, dtype=np.float32)

  image = image.transpose([2, 0, 1])
  
  mean_vec = np.array(mean)
  std_vec = np.array(std)

  for i in range(img_data.shape[0]):
    # Scale each pixel to [0, 1] and normalize per channel.
    img_data[i, :, :] = (image[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

  return img_data


def preprocess_classification(source_dir=None, source_file=None, input_shape=(3,224,224), build_dir='build/data/imagenet', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  '''
    source_path:   define which path the data locate
    source_file:   define which pic in source_path will be used, if not defined source_file, use all
  '''
  if  os.path.exists(build_dir):
    return
  else:
    os.mkdirs(build_dir)

  if not os.path.exists(source_dir):
    logging.error('source dir not existed %s'%source_dir)
  if os.path.exists(source_file):
    imglist = []
    with open(source_path,'r') as f:
      for line in f.readlines():
        imgname = line.strip().split(' ')[0]
        if imgname:
          imglist.append(imgname)
  else:
    imglist = os.path.list(source_dir)
  
  for img in imglist:
    imgname = os.path.join(source_dir,img)
    preprocessed_name = os.path.join(build_dir,img)
    preprocessed_data = preprocess_imagenet_data(imgname, input_shape, mean, std)
    np.save(preprocessed_name, preprocessed_data)
########### calibrator ###########


class classificationEntropyCalibrator(trt.IInt8EntropyCalibrator2):
  def __init__(self, cache_file, cal_files=None, batch_size=8, input_shape=(3,416,416)):
      trt.IInt8EntropyCalibrator2.__init__(self)

      def load_imagenet_data(num_cali,input_shape, cal_files=None):
        imagnet_dir = DataZoo.get('imagenet')
        build_dir = 'build/data/imagenet'

        preprocess_classification(source_dir=imagenet_dir, build_dir=build_dir, \
          input_shape=(3,224,224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if cal_files:
          imglist = []
          with open(cal_files,'r') as f:
            for line in f.readlines():
              imgname = line.strip().split(' ')[0]
              if imgname:
                imglist.append(imgname)
                
        else:
          imglist = os.listdir(build_dir)
        datas = np.random.random((num_cali,input_shape[0],input_shape[1],input_shape[2]))
        for i in range(num_cali):
          img = os.path.join(build_dir, imglist[i])
          data = np.load(img)
          datas[i,:] = data
        return datas

      self.cache_file = cache_file
      self.data = load_imagenet_data(8, input_shape, cal_files=cal_files)
      self.batch_size = batch_size
      self.current_index = 0
      # Allocate enough memory for a whole batch.
      self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

  def get_batch_size(self):
      return self.batch_size

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
      if os.path.exists(self.cache_file):
          print('read cache ',self.cache_file)
          with open(self.cache_file, "rb") as f:
              return f.read()

  def write_calibration_cache(self, cache):
      print('write cache ',cache)
      with open(self.cache_file, "wb") as f:
          f.write(cache)

########### build engine ###########

def check_network(network):
    if not network.num_outputs:
        logging.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    logging.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logging.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logging.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))

def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            print("fbs:{}".format(fbs))
            print("shape:{}".format(shape))
            profile.set_shape(inp.name, min=(1, *shape), opt=(fbs, *shape), max=(fbs, *shape))
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

            profiles[bs].set_shape(inp.name, min=(1, *shape), opt=(bs, *shape), max=(bs, *shape))

    return list(profiles.values())

def add_profiles(config, inputs, opt_profiles):
  logging.debug("=== Optimization Profiles ===")
  for i, profile in enumerate(opt_profiles):
    for inp in inputs:
      _min, _opt, _max = profile.get_shape(inp.name)
      logging.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
  config.add_optimization_profile(profile)

def build_engine(onnx_file_path, engine_file_path="", batch_size=-1, input_shape=(3,416,416), \
  precison='fp16', max_workspace_size=1<<31, calib_cache="calibration_cache", \
  calibrator=classificationEntropyCalibrator):  

  with trt.Builder(TRT_LOGGER) as builder,\
    builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

    logging.info('building batchsize {} input shape {}'.format(batch_size,input_shape))
    builder.max_batch_size = batch_size
    config.max_workspace_size = max_workspace_size # 1<<31 = 2GB
    with open(onnx_file_path, 'rb') as model:
      if not parser.parse(model.read()):
        for error in range(parser.num_errors):
          logging.error(parser.get_error(error))
          return None
    logging.info('Completed parsing of ONNX file')
    check_network(network)    # not neccessory

    if precison == 'int8':   
      config.set_flag(trt.BuilderFlag.INT8)
      config.int8_calibrator = calibrator(cache_file=calib_cache, batch_size=batch_size, input_shape=input_shape)
    elif precison == 'fp16':
      config.set_flag(trt.BuilderFlag.FP16)
    else:
      pass

    batch_sizes = [batch_size]
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
    add_profiles(config, inputs, opt_profiles)
    #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    logginge.info('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

    engine = builder.build_engine(network, config)
    logging.info("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
      f.write(engine.serialize())
    return engine

