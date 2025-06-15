import cv2
import numpy as np

def find_box_from_mask(mask):
    y, x = np.where(mask == 1)
    x0 = x.min()
    x1 = x.max()
    y0 = y.min()
    y1 = y.max()
    return [x0, y0, x1, y1]

def limit_rect(mask, box_ratio):
    """ judge the expanded box if is outside the mask """
    height, width = mask.shape[0], mask.shape[1]
    box = find_box_from_mask(mask)
    w, h = box[2] - box[0], box[3] - box[1]
    w_ratio = w * box_ratio
    h_ratio = h * box_ratio
    x1 = box[0] - w_ratio/2 + w / 2
    y1 = box[1] - h_ratio/2 + h / 2
    x2 = x1 + w_ratio
    y2 = y1 + h_ratio
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= width:
        x2 = width
    if y2 >= height:
        y2 = height
    return x1, y1, x2-x1, y2-y1

def find_center_from_mask_new(mask, box_ratio=4, n_fg=50, n_bg=100):
# def get_all_point_info(mask, box_ratio, n_fg, n_bg):
    """
    input:
        mask:     单个目标的mask
        bg_ratio: 基于最大外接框的基础上扩张bg_ratio倍
        n_fg:     前景点个数
        n_bg:     背景点个数
    Return:
        point_coords(ndarry): size=M*2, 选取M个点(前景或背景)
        point_labels(ndarry): size=M, M个点的Label, 1为前景, 0为背景
    """
    # 找质心
    # M = cv2.moments(mask)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # center_point = np.array([cX, cY]).reshape(1, 2)
    cX = 0
    cY = 0
    center_point = np.array([cX, cY]).reshape(1, 2)

    # 获得前景点
    indices_fg = np.where(mask == 1)
    points_fg = np.column_stack((indices_fg[1], indices_fg[0]))
    after_p_fg = points_fg.copy()
    # 均匀采样n个点
    step_fg = int(len(points_fg) / n_fg)
    # print(len(points_fg))
    num_fg = len(points_fg)
    if step_fg == 0:
        points_fg = points_fg
    else:
        points_fg = points_fg[::step_fg, :]
        if len(points_fg) > n_fg:
            points_fg = points_fg[:n_fg, :]
    
    num_not_enough_one = n_fg - len(points_fg)
    # 找最大外接框
    x, y, w, h = limit_rect(mask, box_ratio)
    box1 = (x, y, x+w, y+h)
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # 获得背景点
    yy, xx = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))
    roi = mask[y:y+h, x:x+w]
    bool_mask = roi == 0
    points_bg = np.column_stack((yy[bool_mask], xx[bool_mask]))
    
    if len(points_bg) == 0 or (len(points_bg) + len(after_p_fg) < n_fg + n_bg):
        indices_bg = np.where(mask == 0)
        points_bg = np.column_stack((indices_bg[1], indices_bg[0]))
    
    # 均匀采样n个点
    step_bg = int(len(points_bg) / n_bg)
    if step_bg == 0:
        num_not_enough_zero = n_bg - len(points_bg)
        step_bg = int(len(after_p_fg) / num_not_enough_zero)
        # print("****")
        # print(len(after_p_fg))
        # print(len(points_bg))
        # print(num_not_enough_zero)
        # print(step_bg)
        points_bg = np.concatenate((after_p_fg[::step_bg, :], points_bg), axis=0)
    else:
        if num_not_enough_one != 0:
            # print("+++")
            # print(num_not_enough_one)
            step_fg = int(len(points_bg) / (num_not_enough_one))
            points_fg = np.concatenate((points_fg ,points_bg[::step_fg, :]), axis=0)
        points_bg = points_bg[::step_bg, :]
    # 获取point_coords
    # points_fg = np.concatenate((center_point, points_fg[1:]), axis=0)
    
    point_coords = np.concatenate((points_fg, points_bg), axis=0)
    point_labels = np.concatenate((np.ones(len(points_fg)), np.zeros(len(points_bg))), axis=0)

    # if flag1 == 1:
    #     point_labels = np.concatenate((np.ones(n_fg), np.ones(n_bg)), axis=0)
    # elif flag2 == 1:
    #     point_labels = np.concatenate((np.zeros(n_fg), np.zeros(n_bg)), axis=0)
    # else:
    #     point_labels = np.concatenate((np.ones(n_fg), np.zeros(n_bg)), axis=0)

    return point_coords, point_labels, points_fg, points_bg, box1, (cX, cY) 

def find_all_info(mask, label_list, point_num):
    point_list = []
    point_label_list = []
    mask_list = []
    box_list = []
    # multi-object processing
    # print("****")
    # print(label_list)
    # print(mask.shape)
    # print("++++")
    for current_label_id in range(len(label_list)):
        
        current_mask = mask[current_label_id]
        current_center_point_list, current_label_list,_,_,_,_=  find_center_from_mask_new(current_mask)
        current_box = find_box_from_mask(current_mask)
        point_list.append(current_center_point_list[0:point_num,:])
        point_label_list.append(current_label_list[0:point_num,])
        box_list.append(current_box)
        # print(point_list)
        # print(point_label_list)
        # print(box_list.shape)
    return np.array(point_list), np.array(point_label_list), np.array(box_list)