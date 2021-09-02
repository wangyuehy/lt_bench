import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class test(unittest.TestCase):
  rel_path = 'cv/object_detection/ssd/ssd_mobilenetv2_torch'
  build_path = os.path.join('build', rel_path)
  def test_get_abs_config(self):
    func_name = 'build_torch'
    from omegaconf import OmegaConf
    from base import get_absolute_config
    base_config = OmegaConf.load(os.path.join(self.rel_path, 'ssd_mobilenetv2.yaml'))
    final_config = get_absolute_config(base_config, func_name)
    print(OmegaConf.to_yaml(final_config))

  def test_torch_load_pretrained(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    net = model.torch(model_path=None, load_pretrained=True)
    self.assertTrue(net)
    self.assertTrue(len(net.backbone.conv1.conv.weight.shape) == 4) # TODO

  def test_torch_load_empty(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    net = model.torch(model_path=None, load_pretrained=False)
    self.assertTrue(net)
    self.assertTrue(len(net.backbone.conv1.conv.weight.shape) == 4)
    
  def test_torch_build_default(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    net = model.torch(load_pretrained=False)
    self.assertTrue(net)
    self.assertTrue(len(net.backbone.conv1.conv.weight.shape) == 4)

  def test_torch_build_specified(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    net = model.torch(model_cfg=os.path.join(self.rel_path, 'ssd_mobilenetv2_CBR_512_200e_coco.py'),
                      model_path=os.path.join(self.rel_path, 'ssd_mobilenetv2_CBR_512_200e_coco_epoch_200.pth'))
    self.assertTrue(net)
    self.assertTrue(len(net.backbone.conv1.conv.weight.shape) == 4)

  def test_process_default(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    model.preprocess(force_preprocess=True)

  def test_process_default2(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    model.preprocess(force_preprocess=False)

  def test_process_one(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    proc_img = model.preprocess_one(input_shape=[3, 512, 512])
    self.assertTrue(list(proc_img.shape) == [3, 512, 512])

  def test_onnx_load_specified(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    model_path = os.path.join(self.rel_path, 'ssd_mobilenetv2_3x512x512_dynamicbatch.onnx')
    self.assertTrue(os.path.isfile(model_path))
    net, model_path_ret = model.onnx(model_path=model_path, return_path=True)
    self.assertTrue(net)
    self.assertTrue(model_path_ret==model_path)

  def test_onnx_load_pretrained(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    model_path = os.path.join(self.build_path, 'test_onnx_load_pretrained.onnx')
    if os.path.isfile(model_path):
      os.remove(model_path)
    self.assertFalse(os.path.isfile(model_path))
    net, model_path_ret = model.onnx(model_path=model_path, load_pretrained=True, return_path=True)
    self.assertTrue(net)
    self.assertTrue(os.path.isfile(model_path_ret))
    self.assertTrue(model_path_ret == model.config.onnx.pretrained.model_path)

  def test_onnx_build(self):
    import base
    model = base.loadNet(task='object_detection', model_type='ssd', backbone='mobilenetv2')()
    net = model.onnx(torch_model_path=os.path.join(self.rel_path, 'ssd_mobilenetv2_CBR_512_200e_coco_epoch_200.pth'),
                    model_path=os.path.join(self.build_path, 'ssd_mobilenetv2_3x512x512_dynamicbatch.onnx'),
                    load_pretrained=False)
    self.assertTrue(net)



if __name__ == '__main__':
  t = test()
  t.test_get_abs_config()
  t.test_torch_load_pretrained()
  t.test_torch_load_empty()
  t.test_torch_build_default()
  t.test_torch_build_specified()
  t.test_process_default()
  t.test_process_default2()
  t.test_process_one()
  t.test_onnx_load_specified()
  t.test_onnx_load_pretrained()
  t.test_onnx_build() # TODO, bug


