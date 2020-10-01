#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json

from libs.configs import cfgs

anno_dir = 'data/{}/annotations'.format(cfgs.DATASET_NAME)

anno_files = [os.path.join(anno_dir, f) for f in os.listdir(
    anno_dir) if f.endswith('json') and not f.startswith('.')]

classes = []

for af in anno_files:
  if not os.path.exists(af):
    continue
  with open(af) as f:
    json_dict = json.load(f)
    cls = json_dict['shapes'][0]['label']

  if not cls:
    continue

  classes.append(cls)

classes = list(set(classes))

with open(os.path.join('data/{}'.format(cfgs.DATASET_NAME), 'classes.txt'), 'w') as f:
  for i, cls in enumerate(classes, 1):
    if 'rotate' in cls:
      continue
    f.write('{}\n'.format(cls))
