# -*-coding:utf-8-*-
import torch
import numpy as np
from numpy import array
from src.lib.utils.iou3d.iou3d_utils import nms_gpu

def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


# IOU计算
# 假设box1维度为[N,4]   box2维度为[M,4]
# M和N代表的行
# box代表的是多个框,一行存储的是左下角的坐标和右上角的坐标(有博主写成了左上角和右下角)
# box1有N个框,box2有M个框
def iou(self, box1, box2):
    N = box1.size(0)
    M = box2.size(0)
    # box1和box2在不同的维度拓展是为了保证每一个框的左下角点和右上角都能够进行比较，注意看例子
    lt = torch.max(  # 左下角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(  # 右上角的点
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # 两个box没有重叠区域，直接去掉
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N, M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N, M)  # (N,M)

    iou = inter / (area1 + area2 - inter)
    return iou

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def numpy_nms(bboxes, scores, threshold=0.5):
    corners = np.array([compute_box_3d(i) for i in bboxes])

    x1 = corners[:, 0, 0]
    y2 = corners[:, 2, 0]
    x2 = corners[:, 0, 2]
    y1 = corners[:, 2, 2]
    areas = (x2 - x1) * (y2 - y1)  # [N,] 每个bbox的面积
    order = np.argsort(scores)[::-1] # 降序排列

    keep = []
    while len(order) > 0:  # torch.numel()返回张量元素个数
        if len(order) == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clip(min=x1[i])  # [N-1,]
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)  # [N-1,]

        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
        idx = (iou <= threshold).nonzero()[0]#.squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
        if len(idx) == 0:
            break
        order = order[idx + 1]  # 修补索引之间的差值
    keep = np.array(keep)
    return keep


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def compute_box_3d(bbox):

    '''
        rect/ref camera coord:
        right x, down y, front z
    '''
    # compute rotational matrix around yaw axis
    R = roty(float(bbox[-1])) #3*3

    l = bbox[3]
    w = bbox[1]
    h = bbox[0]
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corner = np.vstack([x_corners, y_corners, z_corners])
    corner_3d = np.dot(R,corner)
    corner_3d[0,:] = corner_3d[0,:] + bbox[3]
    corner_3d[1,:] = corner_3d[1,:] + bbox[4]
    corner_3d[2,:] = corner_3d[2,:] + bbox[5]

    return corner_3d

def bbox3d2d(bbox_3d, intrinsic):
    bbox_corner = compute_box_3d(bbox_3d)
    bbox2d_corner = np.dot(intrinsic, bbox_corner)[:3, :]
    bbox2d_corner = bbox2d_corner / bbox2d_corner[-1, :]
    min_x = min(bbox2d_corner[0, :])
    min_y = min(bbox2d_corner[1, :])
    max_x = max(bbox2d_corner[0, :])
    max_y = max(bbox2d_corner[1, :])

    return np.array([min_x, min_y, max_x, max_y])

def aug_iou_bbox(bboxes_dict, calib, conf_low=2.0, conf_min=2.0, conf_max=4.0,
                 param1=4.0, param2=6.0, nms_thres=0.5, val=None):
    bboxes = []
    depth_confs = []
    labels = []
    scores = []
    alphas = []

    for bbox_dict in bboxes_dict:
        bbox_dict = bbox_dict.split(' ')
        alpha = bbox_dict[3]
        bbox = np.array([float(i) for i in bbox_dict[4:15]])
        label = bbox_dict[0]
        if bbox_dict[16] == '\n':
            depth_conf = float(bbox_dict[15])
        else:
            depth_conf = float(bbox_dict[16].replace('\n', ''))
        score = float(bbox_dict[15].replace('\n', ''))
        bboxes.append(bbox)
        labels.append(label)
        depth_confs.append(depth_conf)
        scores.append(score)
        alphas.append(alpha)

    bboxes_aug = []
    bboxes_keep = []
    depth_confs_aug = []
    depth_confs_keep = []
    labels_aug = []
    labels_keep = []
    scores_aug = []
    scores_keep = []
    alphas_aug = []
    alphas_keep = []
    drop_num = 0
    mask = 0
    mask_aug = []
    liss = []
    for bbox, label, depth_conf, score, alpha in \
            zip(bboxes, labels, depth_confs, scores, alphas):
        depth_conf = 1.8
        inddd = 1/depth_conf
        if depth_conf < conf_low:
            drop_num +=1
            mask_aug.append(mask)
            mask += 1
            continue
        elif depth_conf >= conf_low and depth_conf < conf_min:
            bboxes_keep.append(bbox)
            depth_confs_keep.append(depth_conf)
            labels_keep.append(label)
            scores_keep.append(score)
            alphas_keep.append(alpha)
            mask_aug.append(mask)
        elif depth_conf >= conf_max:
            bboxes_keep.append(bbox)
            depth_confs_keep.append(depth_conf)
            labels_keep.append(label)
            scores_keep.append(score)
            alphas_keep.append(alpha)
            mask_aug.append(mask)
        elif depth_conf >= conf_min and depth_conf < conf_max:
            depth = np.sqrt(bbox[7] ** 2 + bbox[8] ** 2 + bbox[9] ** 2)
            depth_2d = np.sqrt(bbox[7] ** 2 + bbox[9] ** 2)
            location_alpha1 = bbox[8] / depth
            location_alpha2 = bbox[7] / depth_2d
            line = np.arange(int(param1 / depth_conf) * 2 + 1) - int(param1 / depth_conf)
            if param2 / depth_conf > float(bbox_dict[10]):
                aug_depth = param2 / depth_conf
                depth_conf1 = 1 / (1 + np.exp(-depth_conf))
                param2_1 = aug_depth / (depth*10)
                liss.append(param2_1)
            else:
                aug_depth = float(bbox_dict[10])
            depth_list = line * aug_depth + depth
            a =1
            for i, depth_index in enumerate(depth_list):
                bbox_temp = np.copy(bbox)
                depth_conf_temp = np.copy(depth_conf)
                score_temp = np.copy(score)
                if depth_index == depth:
                    bboxes_aug.append(bbox)
                    depth_confs_aug.append(depth_conf)
                    scores_aug.append(score)
                    labels_aug.append(label)
                    alphas_aug.append(alpha)
                    mask_aug.append(mask)
                    continue
                bbox_temp[8] = depth_index * location_alpha1
                temp = np.sqrt(depth_index ** 2 - bbox_temp[8] ** 2)
                bbox_temp[7] = temp * location_alpha2
                bbox_temp[9] = np.sqrt(temp ** 2 - bbox_temp[7] ** 2)
                depth_conf_temp -= 0.01 * abs(line[i])
                score_temp -= 0.01 * abs(line[i])
                if (-40 < bbox_temp[7]) and (bbox_temp[7] < 40) and (1 < bbox_temp[9]) and (bbox_temp[9] < 70):
                    bbox2d_temp = bbox3d2d(bbox_temp[4:], calib[:3, :3])
                    bbox_temp[:4] = bbox2d_temp
                    bboxes_aug.append(bbox_temp)
                    depth_confs_aug.append(depth_conf_temp)
                    scores_aug.append(score_temp)
                    labels_aug.append(label)
                    alphas_aug.append(alpha)
                    mask_aug.append(mask)
                else:
                    print('stop')
        else:
            raise ValueError
        mask += 1

    bboxes = np.array(bboxes_keep + bboxes_aug)
    depth_confs = np.array(depth_confs_keep + depth_confs_aug)
    labels = np.array(labels_keep + labels_aug)
    scores = np.array(scores_keep + scores_aug)
    alphas = np.array(alphas_keep + alphas_aug)


    # return bboxes, alphas, depth_confs, labels, scores, np.array(mask_aug), drop_num
    if len(bboxes) > 1:
        bboxes_input = np.zeros_like(bboxes[:, 4:])
        bboxes_input[:, :3] = bboxes[:, 7:10] + 100 #[x, y, z, h, w, l, ry]
        bboxes_input[:, 3:6] = bboxes[:, 4:7]
        bboxes_input[:, 6] = bboxes[:, -1]
        bboxes_bev = boxes3d_to_bev_torch(torch.FloatTensor(bboxes_input))
        # nms_results = nms_gpu(bboxes_bev, depth_confs, nms_thres)
        nms_results =[0]
        if len(nms_results) < len(bboxes_bev):
            print('nms: ' + str(val) + '  ' + str(drop_num))
    else:
        nms_results = np.arange(0, len(bboxes))
    return bboxes[nms_results], alphas[nms_results], depth_confs[nms_results], \
               labels[nms_results], scores[nms_results], np.array(mask_aug)[nms_results],\
           drop_num

