# YOLO-Tracking-Distancing
An implementation of target tracking and distance measurement based on dataset from SAIC.

## Installation
Before start, [CUDA v10.0](https://developer.nvidia.com/cuda-10.0-download-archive) and [Cudnn v7.4](https://developer.nvidia.com/rdp/cudnn-archive) are required.

Then install other requirements:
```
pip3 install -r ./docs/requirements.txt
```

## Detection

### dataset
dog/cat dataset provided by SAIC are supposed to be downloaded to ```./data/dataset/dogcat/image/```

Then produce labels from xml files
```
python dataprep_xml2txt.py
```

or you may just use your own dataset with proper label extraction.

### pre-trained model
download YOLOV3 pre-trained model trained on coco for transfer learning
```
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
```

### Then the model is ready to be trained

```
python detection_train.py
```

It's also possible to train from scratch if following configuration is changed in ```./core/config.py```
```
__C.TRAIN.TRAINING_FROM_SCRATCH = False
```

### Test

```
python detection_test.py  # Detection images are saved in `./data/detection`
```
The dog & cat are detected and the distance is shown on the boundingboxes

![image](https://github.com/tanzenyoyoyo/YOLOV3-Tracking-DistanceMeasurement/blob/master/result_demo/fovs1_frame25_out15.png)


Then the model can be evaluated with
```
python mAP/result.py  # evaluation results are saved in `./mAP`
```

## Tracking

For tracking the movement of cat & dog
```
python tracking.py  # Tracking images are saved in `./data/tracking/image`
```

To tracking cat & dog in video, just do following configuration in ```./core/config.py```
```
__C.TRACKING.INPUT_TYPE           = "video"
```

Then the tracking images and video are saved in ```./data/tracking/video```

![image](https://github.com/tanzenyoyoyo/YOLOV3-Tracking-DistanceMeasurement/blob/master/result_demo/Tracking_video.jpg)

## Acknowledgement

This work is builded on many excellence works, which including
- [YunYang1994's yolov3 on tensorflow2.0](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3)
- [mattzheng's keras-yolov3-KF-objectTracking](https://github.com/mattzheng/keras-yolov3-KF-objectTracking)
- Both repositories are referred to [darknet](https://github.com/pjreddie/darknet) either directly or indirectly.



