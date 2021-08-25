import hydra
from omegaconf import DictConfig, OmegaConf
import pdb
import sys
from hydra import compose,initialize
import yaml


@hydra.main(config_path="./", config_name='conf')
def my_app(cfg)-> None:
  print(OmegaConf.to_yaml(cfg))

@hydra.main(config_path='./',config_name='conf')
def real_test(cfg):
  print('\nreal test')
  cfg.db.infile=666

  my_app(cfg)

config_path ='./'
config_name='conf'


@hydra.main(config_path=config_path, config_name=config_name)
def parse_args():
  return cfg


def get_cfg():
  print('\n\nget_cfg')
  print(sys.argv[1:])
  print('upper is the argument')

  with initialize(config_path=config_path, job_name="app"):
    cfg = compose(config_name=config_name, overrides=sys.argv[1:])
    print(OmegaConf.to_yaml(cfg))
  return cfg

class dtest:
  def __init__(self):
    pass

  #@hydra.main(config_path=config_path, config_name=config_name)
  def vtest(self):
    cfg = parse_args(sys.argv)
    print('\nvtest')
    print(OmegaConf.to_yaml(cfg))

    cfg =get_cfg()
    print('\n vtest 2')
    print(OmegaConf.to_yaml(cfg))

def parser():
  for line in sys.argv:
    print( line)

class vtest():
  def __init__(self):
    pass

  def vtest1(self,cfg):
    print('infile', cfg['db']['infile'])

  def test_vtest1(self):
    cfg =get_cfg()
    self.vtest1(cfg)

    dcfg={'db':{'infile':666}}
    self.vtest1(dcfg)




def check_parameter(base_yaml,extend_yaml=None):
    def wrapper(func):
        def innerwrapper(*args, **kwargs):
            base_conf = OmegaConf.load(base_yaml)
            if yaml_file2:
              extend_config = OmegaConf.load(extend_yaml)
              base_conf = OmegaConf.merge(base_conf, extend_config)
            #print(func.__name__)
            config = base_conf.get(func.__name__, [])
            for k in kwargs:
                config[k] = kwargs[k]
            #return func(*args, **config)
            return func(*args, config)
        return innerwrapper
    return wrapper

def decorator2(func):
  def wrapper(class_instance,*args, **kwargs):
    func(class_instance,*args,**kwargs)
    print(class_instance.yaml_file)
    print(class_instance.yaml_file2)
    
  return wrapper

def printyaml(y1,y2=None):
  print(y1)
  print(y2)

class yaml_test:
  def __init__(self):
    self.yaml_file = 'baseModel/test/conf.yaml'
    self.yame_file2 = None
    #printyaml(self.yaml_file, self.yame_file2)
  #@check_parameter(yaml_file=self.yaml_file)
  def test1(self, cfg):
    print('dataset ', cfg['dataset'])
    print(cfg)
    #['dataset'])

  def test2(self,cfg,data='555'):
    #print('dataset ',cfg['dataset'])
    print(cfg)
    print(data)

class yaml_test_child(yaml_test):
  yaml_file2='newyaml'
  @decorator2
  def __init__(self):
    super(yaml_test_child,self).__init__()
    

if __name__ =='__main__':
  #my_app()
  #my_app(db.infile=10000)
  #my_app()
  #real_test()
  #dtest().vtest()
  #parser()
  #vtest().test_vtest1()
  #yaml_test().test2('sd')
  #yaml_test().test2('sd',data=65)
  yaml_test_child()