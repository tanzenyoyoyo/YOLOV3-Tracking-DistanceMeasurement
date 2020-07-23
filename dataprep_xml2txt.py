#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : dataprep_xml2txt.py
#   Author      : Yu Bai, Chunliu Xu, Chengcheng Hou
#   Created date: 2020-07-22
#   Description :
#
#================================================================
import os
import os.path as ops
import xml.etree.ElementTree as ET
import core.utils as utils
from core.config import cfg

new_txt_dir = cfg.DATASET.NEW_TXT_DIR  # dir for txt
old_xml_dir = cfg.DATASET.OLD_XML_DIR  # dir for xml
train_coefficient = cfg.DATASET.TRAIN_COEEFICIENT  # coeff. of training set
image_dir = cfg.DATASET.IMAGE_DIR  #dir for image

if not os.path.exists(new_txt_dir):
    os.makedirs(new_txt_dir)

with open(new_txt_dir+'\\'+'Label_all.txt', 'w+') as f:
    count_all = 0

    assert ops.exists(old_xml_dir), '{:s} not exist'.format(old_xml_dir)
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    for fp in os.listdir(old_xml_dir):

        f.write(' '.join([image_dir + '\\' + fp.split('.')[0] + ".png"]))
        root = ET.parse(os.path.join(old_xml_dir, fp)).getroot()

        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        sz = root.find('size')
        width = float(sz[0].text)
        height = float(sz[1].text)
        filename = root.find('filename').text
        for child in root.findall('object'):  # 找到图片中的所有框

            name = str(child.find('name').text)  # 找到框的标注值并进行读取
            if name == 'cat':
                sub = child.find('bndbox')  # 找到框的标注值并进行读取
                ID = utils.read_class_ids(cfg.YOLO.CLASSES)[name]
                xmin = float(sub[0].text)
                ymin = float(sub[1].text)
                xmax = float(sub[2].text)
                ymax = float(sub[3].text)
                f.write(' '.join(
                [str(' ') + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + str(ID)]))
            elif name == 'dog':
                sub = child.find('bndbox')  # 找到框的标注值并进行读取
                ID = utils.read_class_ids(cfg.YOLO.CLASSES)[name]
                xmin = float(sub[0].text)
                ymin = float(sub[1].text)
                xmax = float(sub[2].text)
                ymax = float(sub[3].text)
                f.write(' '.join(
                    [str(' ') + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + str(ID)]))
            else:
                continue
        f.write('\n')
        count_all += 1
    count_train = int(count_all * train_coefficient)
    count_test = count_all - count_train

# divide dataset into train and test
with open(new_txt_dir+'\\'+'Label_all.txt', 'r') as f:
    line=f.readlines()
    data_train, data_test = [],[]
    for i,rows in enumerate(line):
        if i<count_train:
            data_train.append(rows)
        else:
            data_test.append(rows)

    with open(new_txt_dir+'\\'+'Label_train.txt', 'w') as f_train:
        for i in range(len(data_train)):
            f_train.writelines(data_train[i])
    with open(new_txt_dir + '\\' + 'Label_test.txt', 'w') as f_test:
        for i in range(len(data_test)):
            f_test.writelines(data_test[i])

    print('Down', '\n', 'Num of training image：{}'.format(count_train),'\n',
          'Num of test image：{}'.format(count_test),'\n',
          'overall：{}'.format(count_all))



