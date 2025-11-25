from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from detectors.RTS3D_infer import RTS3DDetector
from opts import opts
import shutil
import torch
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net']

def demo(opt):
    import glob
    import os.path

    from src.lib.utils.aug_iou_bbox import aug_iou_bbox
    import numpy as np
    # 
    conf_low = 0.0
    conf_min = 1.5
    conf_max = 6.0
    nms_thres = 0.5
    param1 = 4.0  # roi_num
    param2 = 12.0  # roi_range
    mono_path = './kitti_format/data/kitti/test_results/momocon_depth_lr_0.3/img_bbox/'
    with open('./kitti_format/data/kitti/test.txt', 'r') as f:
        val_list = f.readlines()

    save_folder = '/home/liux/code/image/stereo/RTS3d_ori2/kitti_format/data/kitti/test_results' \
                  '/momocon_depth_lr_0.3_car_val_%3.1f_%3.1f_%3.1f_%3.1f_%3.1f_%3.1f' \
                  % (conf_low, conf_min, conf_max, param1, param2, nms_thres)

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
            calib = np.array([float(i) for i in calib]).reshape((3, 4))
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
                    kitti_str = str(label_aug) + ' -1.00 -1 ' + str(float(alpha)) + ' ' + \
                                bbox_str + str(np.around(score_aug, 3)) + ' ' + '\n'
                    f.write(kitti_str)
    print('drop sum = ' + str(dropsum))

    opt.debug = max(opt.debug, 1)
    opt.faster=False
    Detector = RTS3DDetector
    detector = Detector(opt)
    if os.path.exists(opt.demo_results_dir):
        shutil.rmtree(opt.demo_results_dir, True)
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      if opt.demo[-3:]=='txt':
          ls = os.listdir(opt.mono_path)
          image_l_names=[os.path.join(opt.data_dir+'/kitti/image/',img[:6]+'.png') for img in ls]
          image_r_names = [os.path.join(opt.data_dir + '/kitti/image/', "{:06d}".format(int(float(img[:6])+7481))+'.png') for img in ls]
          mono_est= [os.path.join(opt.mono_path,img[:6]+'.txt') for img in ls]
      else:
        image_names = [opt.demo]
    time_tol = 0
    num = 0
    os.makedirs(opt.demo_results_dir)

    for (image_name_l,image_name_r,mono_est) in zip(image_l_names,image_r_names,mono_est):
      num+=1
      ret = detector.run(image_name_l,image_name_r,mono_est, opt.nms)
      # torch.onnx.export(detector, image_name_l, '/home/liux/code/image/stereo/'
      #                                           'RTS3d_ori2/model_10_10_10.onnx',
      #                   opset_version=12,
      #                   input_names=['img'],
      #                   output_names=['flow_up'])

      time_str = ''
      for stat in time_stats:
          time_tol=time_tol+ret[stat]
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
      print(time_str + '   |' + str(num))
      # print('save path: ' + opt.demo_results_dir)
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
