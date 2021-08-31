import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class test(unittest.TestCase):
  def test_get_abs_config(self):
    func_name = 'build_torch'
    from omegaconf import OmegaConf
    from base import get_absolute_config
    base_config = OmegaConf.load('cv/object_detection/ssd/ssd_mobilenetv2_torch/ssd_mobilenetv2.yaml')
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
    net = model.torch(model_cfg='cv/object_detection/ssd/ssd_mobilenetv2_torch/ssd_mobilenetv2_CBR_512_200e_coco.py',
                      model_path='cv/object_detection/ssd/ssd_mobilenetv2_torch/ssd_mobilenetv2_CBR_512_200e_coco_epoch_200.pth')
    self.assertTrue(net)
    self.assertTrue(len(net.backbone.conv1.conv.weight.shape) == 4)


if __name__ == '__main__':
  t = test()
  t.test_get_abs_config()
  t.test_torch_load_pretrained()
  t.test_torch_load_empty()
  t.test_torch_build_default()
  t.test_torch_build_specified()


