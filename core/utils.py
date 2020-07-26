#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : utils.py
#   Author      : Yu Bai, Chunliu Xu, Chengcheng Hou
#   Created date: 2020-07-22
#   Description :
#
#================================================================

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import os
from core.config import cfg


def load_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

#    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def read_class_ids(class_file_name):
    '''loads class id from a file'''
    ids = {}
    with open(class_file_name, 'r') as data:
        for id, name in enumerate(data):
            ids[name.strip('\n')] = id
    return ids

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)


        if show_label:
            distance = cal_distance((c1[0] + c2[0])/2, int(c2[1]))
            bbox_mess = '%s: %.2f  distance: %.3f' % (classes[class_ind], float(score), distance)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def calc_center(coor, class_ind, scores, score_limit):
    outboxes_filter15 = []
    outboxes_filter16 = []
    for x, y, z in zip(coor, class_ind, scores):
        if float(z) > score_limit:
            if y == 15:
                outboxes_filter15.append(x)
            if y == 16:
                outboxes_filter16.append(x)

    centers15 = []
    number15 = len(outboxes_filter15)
    for box in outboxes_filter15:
        top, left, bottom, right = box
        center = np.array([[(int(top) + int(bottom)) // 2], [(int(left) + int(right)) // 2]])
        centers15.append(center)
    centers16 = []
    number16 = len(outboxes_filter16)
    for box in outboxes_filter16:
        top, left, bottom, right = box
        center = np.array([[(int(top) + int(bottom)) // 2], [(int(left) + int(right)) // 2]])
        centers16.append(center)
    return centers15, number15, centers16, number16


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    # colors = [(255,99,71) if c==(255,0,0) else c for c in colors ]  # 单独修正颜色，可去除
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def trackerDetection(tracker, image, centers, number, max_point_distance, max_colors=20, track_id_size=0.8):
    '''
        - max_point_distance为两个点之间的欧式距离不能超过30
            - 有多条轨迹,tracker.tracks;
            - 每条轨迹有多个点,tracker.tracks[i].trace
        - max_colors,最大颜色数量
        - track_id_size,轨迹条数
    '''
    # track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    #            (0, 255, 255), (255, 0, 255), (255, 127, 255),
    #            (127, 0, 255), (127, 0, 127)]
    track_colors = get_colors_for_classes(max_colors)

    result = np.asarray(image)

    if (len(centers) > 0):
        # Track object using Kalman Filter
        tracker.Update(centers)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            # 多个轨迹
            if (len(tracker.tracks[i].trace) > 1):
                x0, y0 = tracker.tracks[i].trace[-1][0][0], tracker.tracks[i].trace[-1][1][0]
                cv2.putText(result, str(tracker.tracks[i].track_id), (int(x0), int(y0)), font, track_id_size,
                            (255, 255, 255), 4)
                # (image,text,(x,y),font,size,color,粗细)
                for j in range(len(tracker.tracks[i].trace) - 1):
                    # 每条轨迹的每个点
                    # Draw trace line
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j + 1][0][0]
                    y2 = tracker.tracks[i].trace[j + 1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    if distance < max_point_distance:
                        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 4)
    return tracker, result

def tracking_annotation(image, tracker15, tracker16, INPUT_SIZE, model, CLASSES):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Predict Process
    image_size = image.shape[:2]
    image_data = image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
    bboxes = nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
    out_boxes = []
    out_classes = []
    out_scores = []
    bboxes_new = []
    for bbox in bboxes:
        class_ind = int(bbox[5])
        score = float(bbox[4])
        #score = '%.4f' % score
        if class_ind in [15, 16]:
            if score >= 0.5:
                bboxes_new.append(bbox)

    for bbox in bboxes_new:
        #class_ind = int(bbox[5])
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        out_classes.append(class_ind)
        class_name = CLASSES[class_ind]
        score = '%.4f' % score
        # only for print to check out
        xmin, ymin, xmax, ymax = list(map(str, coor))
        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
        print('\t' + str(bbox_mess).strip())

        out_scores.append(score)
        output_position = list(map(str, coor))
        out_boxes.append(output_position)

    image = draw_bbox(image, bboxes_new)
    centers15, number15, centers16, number16, = calc_center(out_boxes, out_classes, out_scores, score_limit=0.5)
    tracker15, result_cat = trackerDetection(tracker15, image, centers15, number15, max_point_distance=40)
    cv2.putText(result_cat, 'cat: ' + str(number15), (20, 40), font, 1, (0, 0, 255), 2)  # 左上角，猫计数

    tracker16, result_dogcat = trackerDetection(tracker16, result_cat, centers16, number16, max_point_distance=40)
    cv2.putText(result_dogcat, 'dog: ' + str(number16), (20, 80), font, 1, (0, 0, 255), 2)  # 左上角，狗计数

    return result_dogcat


'''
    辅助函数 
    图像文件夹直接变为视频并保存
'''
# 图像 -> 视频
def get_file_names(search_path):
    for (dirpath, _, filenames) in os.walk(search_path):
        for filename in filenames:
            yield filename  # os.path.join(dirpath, filename)


def save_to_video(output_path, output_video_file, frame_rate):
    list_files = sorted([int(i.split('_')[-1].split('.')[0]) for i in get_file_names(output_path)])
    # 拿一张图片确认宽高
    img0 = cv2.imread(os.path.join(output_path, '%s.jpg' % list_files[0]))
    # print(img0)
    height, width, layers = img0.shape
    # 视频保存初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    # 核心，保存的东西
    for f in list_files:
        f = '%s.jpg' % f
        # print("saving..." + f)
        img = cv2.imread(os.path.join(output_path, f))
        videowriter.write(img)
    videowriter.release()
    cv2.destroyAllWindows()
    print('Success save %s!' % output_video_file)
    pass

def cal_distance(u, v):
    dx = cfg.DISTANCE.D_X
    dy = cfg.DISTANCE.D_Y
    u0 = cfg.DISTANCE.U_0
    v0 = cfg.DISTANCE.V_0
    f = cfg.DISTANCE.F
    theta = cfg.DISTANCE.THETA
    alpha = cfg.DISTANCE.ALPHA
    gamma = cfg.DISTANCE.GAMMA
    camera_height = cfg.DISTANCE.CAMERA_HEIGHT

    Rx = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -1 * np.sin(theta), np.cos(theta)]])
    Ry = np.array([[np.cos(alpha), 0, -1 * np.sin(alpha)], [0, 1, 0], [np.sin(alpha), 0, np.cos(alpha)]])
    Rz = np.array([[np.cos(gamma), np.sin(gamma), 0], [-1 * np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R = np.matmul(Rx, Ry, Rz)


    T = np.array([[0], [0], [-camera_height]])

    pixel2image = np.array([[1 / dx, 0, u0],
                            [0, 1 / dy, v0],
                            [0, 0, 1]])
    image2camera = np.array([[f, 0, 0, 0],
                             [0, f, 0, 0],
                             [0, 0, 1, 0]])
    pixel_cor = np.array([[u], [v], [1]])

    M = np.matmul(pixel2image, image2camera)[:, 0: len(np.matmul(pixel2image, image2camera))]
    Mat1 = np.matmul(np.linalg.inv(R), np.linalg.inv(M))
    Mat1 = np.matmul(Mat1, pixel_cor)
    Mat2 = np.matmul(np.linalg.inv(R), T)

    distance_in_world = Mat1[1] * Mat2[2] / Mat1[2] - Mat2[1]
    return distance_in_world
