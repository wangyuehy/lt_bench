import logging
from omegaconf import OmegaConf,DictConfig
import pdb
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")



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

def copy_from_other_node(config, item):
  inherit_src = item.get('inherit',None)
  if inherit_src:
    src_item = OmegaConf.select(config,inherit_src)
    src_item = copy_from_other_node(config, src_item)
    new_item = OmegaConf.merge(config[item], src_item)
  else:
    new_item = config.get(item,[])
  return new_item

def check_parameter(base_yaml=None,extend_yaml=None):
    def wrapper(func):
        def innerwrapper(*args, **kwargs):
            base_conf = OmegaConf.load(base_yaml)
            if extend_yaml:
                extend_config = OmegaConf.load(extend_yaml)
                base_conf = OmegaConf.merge(base_conf, extend_config)
            else:
                raise TypeError('Illegal basel_yaml type')
            config = base_conf.get(func.__name__, [])
            #config = copy_from_other_node(base_conf, func.__name__)
            # to do : add a copy attr which support copy from other node
            for k in kwargs:
                if k not in config:
                  raise TypeError
                config[k] = kwargs[k]
            #return func(*args, **config)
            return func(*args, config)
        return innerwrapper
    return wrapper

def depict_config(cfg):
  print(OmegaConf.to_yaml(cfg))