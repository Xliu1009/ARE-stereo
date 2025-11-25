# -*-coding:utf-8-*-

from __future__ import print_function

import os
import sys
import cv2
import os.path
import torch
import pickle

from PIL import Image
from mmdet3d.datasets.kitti_stereo_dataset_monocon_RTS3d import KittiMonoStereoDatasetMonoCon
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from kitti_object import *
data_root = '/home/liux/code/image/mono/MonoCon/mmdetection3d-0.14.0/data/kitti/'

proposal_3d_file = data_root + 'monocon_0_backbone_rpn_car_val.pkl'
ann_file=data_root + 'kitti_infos_train_mono3d.coco.json'
classes = ['Car'],
save_path = '/home/liux/code/image/mono/MonoCon/mmdetection3d-0.14.0/work_dirs/demo_vis/rpn_ori/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


def visualization():
    # import mayavi.mlab as mlab
    dataset = kitti_object('/home/liux/code/image/mono/MonoCon/'
                           'mmdetection3d-0.14.0/data/kitti',split='val') # linux 路径
    with open(proposal_3d_file, 'rb') as f:
        rpn_results = pickle.load(f)
    f.close()


    val_txt = '/home/liux/data_set/KITTI/kitti_3d/ImageSets/val.txt'
    idx_list = []
    with open(val_txt, 'r') as fh:
        for line in fh:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = int(line) # 以空格为分隔符 将字符串分成
            idx_list.append(words)  # imgs中包含有图像路径和标签

    for i, data_idx in enumerate(idx_list):
        print(str(i) + '/' + str(len(idx_list)))
        proposal = rpn_results[i]['img_bbox']['boxes_3d']
        depth_confs = np.array(rpn_results[i]['depth_confs'])
        # 1-加载标签数据
        objects = dataset.get_label_objects(data_idx)
        # print("There are %d objects.", len(objects))
        save_path_temp = save_path + f'%06d' %data_idx

        # 2-加载图像
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape

        # 3-加载点云数据
        pc_velo = dataset.get_lidar(data_idx)[:, 0:3]  # (x, y, z)

        # 4-加载标定参数
        calib = dataset.get_calibration(data_idx)

        # # 5-可视化原始图像
        # print(' ------------ show raw image -------- ')
        # Image.fromarray(img).show()

        # 6-在图像中画2D框
        # print(' ------------ show image with 2D bounding box -------- ')
        # show_image_with_boxes(img, objects, calib, data_idx, False)
        #
        # 7-在图像中画3D框
        # print(' ------------ show image with 3D bounding box ------- ')
        # show_image_with_boxes(img, objects, calib, save_path_temp, True)
        #
        # # 8-将点云数据投影到图像
        # print(' ----------- LiDAR points projected to image plane -- ')
        # show_lidar_on_image(pc_velo, img, calib, data_idx, img_width, img_height)

        # 9-画BEV图
        # print('------------------ BEV of LiDAR points -----------------------------')
        show_lidar_topview(pc_velo, objects, calib, save_path_temp)

        # 10-在BEV图中画2D框
        # print('--------------- BEV of LiDAR points with bobes ---------------------')
        img1 = cv2.imread(save_path_temp + '_bev.png')
        show_lidar_topview_with_boxes(img1, objects, calib, save_path_temp)


        # print('--------------- BEV of LiDAR points with bobes ---------------------')
        if len(proposal) > 0:
            img1 = cv2.imread(save_path_temp + '_bev.png')
            show_lidar_topview_with_proposals(img1, proposal, calib, save_path_temp, depth_confs)

if __name__ == '__main__':
    visualization()