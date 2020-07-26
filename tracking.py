#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : Tracking.py
#   Author      : Yu Bai, Chunliu Xu, Chengcheng Hou
#   Created date: 2020-07-22
#   Description :
#
#================================================================
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode
import cv2
import os
import shutil
from core.KalmanFilterTracker import Tracker  # 加载卡尔曼滤波函数

detection_dir_path_video = cfg.TRACKING.OUTPUT_VIDEO_PATH
detection_dir_path_image = cfg.TRACKING.OUTPUT_IMAGE_PATH
if cfg.TRACKING.INPUT_TYPE == "video":
    if os.path.exists(detection_dir_path_video): shutil.rmtree(detection_dir_path_video)
    os.makedirs(detection_dir_path_video)
else:
    if os.path.exists(detection_dir_path_image): shutil.rmtree(detection_dir_path_image)
    os.makedirs(detection_dir_path_image)

INPUT_SIZE = 416
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

# Build Model
input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
if '.weights' in cfg.TRACKING.WEIGHTS:
    utils.load_weights(model, cfg.TRACKING.WEIGHTS)
else:
    model.load_weights(cfg.TRACKING.WEIGHTS)


# tracking
tracker15 = Tracker(cfg.TRACKING.DIST_THRESH,
                    cfg.TRACKING.MAX_FRAMES_TO_SKIP,
                    cfg.TRACKING.MAX_TRACE_LENGHT,
                    cfg.TRACKING.TRACKIDCOUNT)
tracker16 = Tracker(cfg.TRACKING.DIST_THRESH,
                    cfg.TRACKING.MAX_FRAMES_TO_SKIP,
                    cfg.TRACKING.MAX_TRACE_LENGHT,
                    cfg.TRACKING.TRACKIDCOUNT * 2)

# video as input
if cfg.TRACKING.INPUT_TYPE == "video":
    path = cfg.TRACKING.INPUT_VIDEO_PATH + cfg.TRACKING.VIDEO_NAME
    vid = cv2.VideoCapture(path)
    n = 0
    while (True):
        return_value, frame = vid.read()
        if return_value:
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame
        else:
            break
    #        raise ValueError("No image!")
        result_dogcat = utils.tracking_annotation(image, tracker15, tracker16, INPUT_SIZE, model, CLASSES)
        cv2.imwrite(detection_dir_path_video + '/%s.jpg' % n, result_dogcat, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        n += 1


    # image to video
    output_path = os.path.join(detection_dir_path_video, '')  # 输入图片存放位置
    output_video_file = detection_dir_path_video + cfg.TRACKING.VIDEO_NAME # 输入视频保存位置以及视频名称
    utils.save_to_video(detection_dir_path_video, output_video_file, 20)

# image as input
if cfg.TRACKING.INPUT_TYPE == "image":
    with open(cfg.TRACKING.INPUT_IMAGE_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('\\')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result_dogcat = utils.tracking_annotation(image, tracker15, tracker16, INPUT_SIZE, model, CLASSES)
            path = detection_dir_path_image + "/{}.jpg".format(image_name)
            cv2.imwrite(path, result_dogcat)

print('Down!')






