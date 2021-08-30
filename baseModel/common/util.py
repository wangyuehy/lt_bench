import logging
from omegaconf import OmegaConf,DictConfig
import pdb
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

log = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
log.addHandler(stdout_handler)

def dict_get(d, key, default=None):
    """Return non-None value for key from dict. Use default if necessary."""

    val = d.get(key, default)
    return default if val is None else val

def printItem(item,level=0):
    '''
    ##  [
    #1    [
    #2        
    #3    ],
    ##  ]
    ##  {
    ##      k1:{
                    k2:
                        {


                        }
                }
    ##  }
    '''
    if level == 0:
        print('*'*21)
    if type(item) == dict:
        print(' '*2*level+'{')
        for k in item:
            print(' '*2*level,k,':')
            printItem(item[k],level+1)
        print(' '*2*level+'}')
        return
    if type(item) == list:
        print(' '*2*level +'[')
        for i in item:
            printItem(i, level+1)
        print(' '*2*level+']')
        return
    print(' '*2*level,item)

def get_absolute_config(config, item_key):
  item  = OmegaConf.select(config, item_key)
  inherit_src = item.get('inherit',None)
  if inherit_src:
    inherit_item = get_absolute_config(config, inherit_src)
    return OmegaConf.merge(inherit_item, item)
  else:
    return item

def check_parameter(base_yaml,extend_yaml=None):
    def wrapper(func):
        def innerwrapper(*args, **kwargs):
            base_conf = OmegaConf.load(base_yaml)
            if extend_yaml:
                extend_config = OmegaConf.load(extend_yaml)
                base_conf = OmegaConf.merge(base_conf, extend_config)
            #config = base_conf.get(func.__name__, [])
            config = get_absolute_config(base_conf, func.__name__)
            for k in kwargs:
                if k not in config:
                  raise TypeError("Invalid paramter {}".format(k))
                config[k] = kwargs[k]
            return func(*args, config)
        return innerwrapper
    return wrapper

def depict_config(cfg):
  print(OmegaConf.to_yaml(cfg))

def set_dir(model_path):
  #dirname = os.path.dirname(model_path)
  p = Path(model_path)
  if not p.is_dir():
    p.mkdir(parents=True)

def set_file_dir(model_path):
  dirname = os.path.dirname(model_path)
  p = Path(dirname)
  if not p.is_dir():
    p.mkdir(parents=True)
