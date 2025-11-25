# -*-coding:utf-8-*-

import glob
import os.path

from src.lib.utils.aug_iou_bbox import aug_iou_bbox
import numpy as np

conf_low = 0.0
conf_min = 1.5
conf_max = 6.0
nms_thres = 0.5
param1 = 4.0   # roi_num
param2 = 12.0   # roi_range
mono_path = './kitti_format/data/kitti/test_results/momocon_depth_lr_0.3/img_bbox/'
with open('./kitti_format/data/kitti/test.txt', 'r') as f:
    val_list = f.readlines()

save_folder = '/home/liux/code/image/stereo/RTS3d_ori2/kitti_format/data/kitti/test_results' \
              '/momocon_depth_lr_0.3_car_val_%3.1f_%3.1f_%3.1f_%3.1f_%3.1f_%3.1f' \
              %(conf_low, conf_min, conf_max, param1, param2, nms_thres)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
label_dict = {0: 'Car'}
dropsum = 0
for val in val_list:
    val = val.replace('\n', '')
    mono = mono_path + val + '.txt'
    calib = '/home/liux/data_set/KITTI/kitti_3d/testing/calib/' + val + '.txt'
    with open(calib, 'r') as f:
        calib = f.readlines()[2].replace('\n', '').split(' ')[1:]
        calib = np.array([float(i) for i in calib]).reshape((3,4))
    if os.path.exists(mono):
        with open(mono, 'r') as f:
            dt = f.readlines()
    else:
        continue
    with open(save_folder + '/' + val.replace('\n', '') + '.txt', 'w') as f:
        if len(dt) > 0:
            bboxes_aug, alphas, depth_confs_aug, labels, scores_aug, mask_aug, drop_num = \
                aug_iou_bbox(dt, calib, conf_low, conf_min, conf_max, param1, param2, nms_thres, val)
            dropsum += drop_num
            if len(bboxes_aug) > 25:
                print(val + " >25")
            for bbox_aug, alpha, depth_conf_aug, label_aug, score_aug, mask in \
                    zip(bboxes_aug, alphas, depth_confs_aug, labels, scores_aug, mask_aug):
                bbox_str = ''
                for i in bbox_aug:
                    bbox_str = bbox_str + str(i) + ' '
                kitti_str = str(label_aug) +' -1.00 -1 ' + str(float(alpha)) + ' ' + \
                            bbox_str + str(np.around(score_aug, 3)) + ' ' + '\n'
                f.write(kitti_str)
print('drop sum = '+ str(dropsum))


