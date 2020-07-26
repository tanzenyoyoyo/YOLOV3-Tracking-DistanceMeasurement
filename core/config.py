#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : config.py
#   Author      : Yu Bai, Chunliu Xu, Chengcheng Hou
#   Created date: 2020-07-22
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# generate dataset label options
__C.DATASET                   = edict()

__C.DATASET.OLD_XML_DIR       = "./data/dataset/dogcat/Annotation_merge"    #数据集的原始标签文件，xml格式
__C.DATASET.NEW_TXT_DIR       = "./data/dataset/dogcat"                     #转换后的数据集的标签文件，txt格式
__C.DATASET.TRAIN_COEEFICIENT = 0.8                                         #训练集占总数据集的比例
__C.DATASET.IMAGE_DIR         = "./data/dataset/dogcat/image"               #数据集图片的文件夹路径


# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/dogcat/Label_train.txt"
__C.TRAIN.BATCH_SIZE          = 4
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-4
__C.TRAIN.LR_END              = 1e-7
__C.TRAIN.WARMUP_EPOCHS       = 1
__C.TRAIN.EPOCHS              = 2
__C.TRAIN.LOG_DIR             = "./data/log"
__C.TRAIN.LOAD_WEIGHTS_PATH   = "./weights/yolov3.weights"      #若权重为.weights文件，则为'./weights/name.weights'
__C.TRAIN.SAVE_WEIGHTS_PATH   = "./weights/yolov3"
__C.TRAIN.TRAINING_FROM_SCRATCH= False                          #transfer learning or training from scratch


# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/dogcat/Label_test.txt"
__C.TEST.PREDICTED_DIR_PATH   = './mAP/predicted'               #txt文件，包含检测的实例及其bounding box的坐标
__C.TEST.GT_DIR_PATH          = './mAP/ground-truth'
__C.TEST.BATCH_SIZE           = 1
__C.TEST.INPUT_SIZE           = 544
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"             #图片，包含检测的实例及其bounding box的坐标
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.45
__C.TEST.LOAD_WEIGHTS_PATH   = "./weights/yolov3"


#Tracking options
__C.TRACKING                      = edict()

__C.TRACKING.INPUT_TYPE           = "video"                     #["image", "video"]
__C.TRACKING.INPUT_VIDEO_PATH     = "./docs"
__C.TRACKING.OUTPUT_VIDEO_PATH    = "./data/tracking/video"
__C.TRACKING.VIDEO_NAME           = '/V011_front60.mp4'
__C.TRACKING.INPUT_IMAGE_PATH     = "./data/dataset/dogcat/Label_test.txt"
__C.TRACKING.OUTPUT_IMAGE_PATH    = "./data/tracking/image"
__C.TRACKING.WEIGHTS              = "./weights/yolov3"
__C.TRACKING.DIST_THRESH          = 180                         #距离阈值，超过阈值时，将删除轨迹并创建新轨迹
__C.TRACKING.MAX_FRAMES_TO_SKIP   = 5                           #超过多少帧没有识别，就放弃识别该物体，未检测到的跟踪对象允许跳过的最大帧数
__C.TRACKING.MAX_TRACE_LENGHT     = 15                          #轨迹的最大长度
__C.TRACKING.TRACKIDCOUNT         = 100                         #每个跟踪对象的标识基础（在此基础上累加）


#Distance options
__C.DISTANCE                      = edict()

#  camera parameters
__C.DISTANCE.D_X                  = 1
__C.DISTANCE.D_Y                  = 1
__C.DISTANCE.U_0                  = 960
__C.DISTANCE.V_0                  = 1208
__C.DISTANCE.F                    = 50
__C.DISTANCE.THETA                = 0
__C.DISTANCE.ALPHA                = 0
__C.DISTANCE.GAMMA                = 0
__C.DISTANCE.CAMERA_HEIGHT        = 1


