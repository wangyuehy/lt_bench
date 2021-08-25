import os
import logging

DATAZOODIR = '/workspace/public/datazoo'

DATADIR={
  'imagenet':os.path.join(DATAZOODIR, 'cv/imagnet'),
  'coco': os.path.join(DATAZOODIR,'cv/coco'),
}

def getDataDir(dataset='imagenet', target_dir=None):
  '''
    given a dataset, return targe dir path
    if defined target dir, return target_dir path
  '''
  if dataset in DATADIR:
    return DATADIR[dataset]
  else:
    logging.error('Invalid dataset naem ',dataset)
