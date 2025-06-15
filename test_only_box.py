from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import os
from copy import deepcopy
import glob
from tqdm import tqdm
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from collections import Counter
import json
from PIL import Image
from skimage import io
join = os.path.join
from utils.get_prompts import find_all_info
import torch.nn.functional as F

def upsample_patch(patch, size=(1024, 1024), mode='nearest', align_corners=None, convert_to_rgb=True):
    """使用 torch.nn.functional.interpolate 上采样图像补丁，并可将单通道转为三通道"""
    # 确保输入是 PyTorch 张量
    if isinstance(patch, np.ndarray):
        patch = torch.from_numpy(patch).float()  # 转换为 float 张量

    # 记录原始维度
    original_dim = patch.dim()
    original_channels = patch.shape[0] if original_dim >= 3 else 1

    # 确保输入是4D张量 [B,C,H,W]
    if patch.dim() == 2:  # 单通道图像 [H,W]
        patch = patch.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        mode = 'nearest'  # 单通道默认使用最近邻插值
    elif patch.dim() == 3:  # 多通道图像 [C,H,W]
        patch = patch.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

                # 上采样
    if patch.shape[1] > 1 and mode != 'nearest':  # 多通道使用双线性插值
        upsampled = F.interpolate(patch, size=size, mode=mode, align_corners=align_corners)
    else:  # 单通道使用最近邻插值
        upsampled = F.interpolate(patch, size=size, mode='nearest')

                # 将单通道转为三通道
    if convert_to_rgb and upsampled.shape[1] == 1:
        upsampled = upsampled.repeat(1, 3, 1, 1)  # 复制通道 [B,1,H,W] -> [B,3,H,W]

                # 恢复原始维度
    if upsampled.shape[0] == 1:
        upsampled = upsampled.squeeze(0)  # [1,C,H,W] -> [C,H,W]

    # 如果原始是单通道且不需要RGB，恢复为 [H,W] 格式
    if original_dim == 2 and not convert_to_rgb:
        upsampled = upsampled.squeeze(0)  # [C,H,W] -> [H,W]
    return upsampled
def process(gt_name: str, image_name: str, mode: str):
    if image_name == None:
        image_name = gt_name.split(".")[0] + args.img_name_suffix
    if mode == "train":
        gt_data = io.imread(join(args.gt_path, gt_name)) # H, W
    elif mode == "valid":
        gt_data = io.imread(join(args.gt_path.replace("train", "valid"), gt_name))
    else:
        gt_path = f"{args.gt_path}/labels"
        gt_data = io.imread(join(gt_path, gt_name))
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truth image
    # resize_gt = sam_transform.apply_image(gt_data, interpolation=InterpolationMode.NEAREST) # ResizeLong (resized_h, 1024)
    # gt_data = sam_model.preprocess_for_gt(resize_gt)

    # exclude tiny objects (considering multi-object)
#     gt = gt_data.copy()
#     label_list = np.unique(gt_data)[1:]
#     del_lab = [] # for check
#     for label in label_list:
#         gt_single = (gt_data == label) + 0
#         if np.sum(gt_single) <= 50:
#             gt[gt == label] = 0
#             del_lab.append(label)
#     assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0])
    
def show_points(image, coords, labels):
    ret_img = deepcopy(image)
    if labels == 1:
        cv2.circle(ret_img, (coords[0][0],coords[0][1]), radius=5,  color=(0,255,0), thickness=-1)
    else:
        cv2.circle(ret_img, (coords[0][0],coords[0][1]), radius=5, color=(0,0,255),  thickness=-1)
    return ret_img

    # calculate dice
def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def limit_rect(mask, box_ratio):
    """ check if the enlarged bounding box extends beyond the image. """
    height, width = mask.shape[0], mask.shape[1]
    # maximum bounding box
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

    # find barycenter
def find_center_from_mask(mask):
    # calculate moments of binary image
    M = cv2.moments(mask)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY], 1

def find_center_from_mask_new(mask, box_ratio=2, n_fg=5, n_bg=5):
# def get_all_point_info(mask, box_ratio, n_fg, n_bg):
    """
    input:
        mask:     single mask
        bg_ratio: expand by a factor of bg_ratio based on the maximum bounding box
        n_fg:     foreground points number
        n_bg:     background points number
    Return:
        point_coords(ndarry): size=M*2, select M points(foreground or background)
        point_labels(ndarry): size=M 
    """
    # find barycenter
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center_point = np.array([cX, cY]).reshape(1, 2)

    # get foreground points
    indices_fg = np.where(mask == 1)
    points_fg = np.column_stack((indices_fg[1], indices_fg[0]))

    # uniformly sample n points
    step_fg = int(len(points_fg) / n_fg)
    # print(len(points_fg))
    points_fg = points_fg[::step_fg, :]
    

    # find the maximum bounding box
    x, y, w, h = limit_rect(mask, box_ratio)
    box1 = (x, y, x+w, y+h)
    x, y, w, h = int(x), int(y), int(w), int(h)

    # get background points
    yy, xx = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))
    roi = mask[y:y+h, x:x+w]
    bool_mask = roi == 0
    points_bg = np.column_stack((yy[bool_mask], xx[bool_mask]))

    # uniformly sample n points
    step_bg = int(len(points_bg) / n_bg)
    points_bg = points_bg[::step_bg, :]

    # get point_coords
    points_fg = np.concatenate((center_point, points_fg[1:]), axis=0)
    point_coords = np.concatenate((points_fg, points_bg), axis=0)
    point_labels = np.concatenate((np.ones(n_fg), np.zeros(n_bg)), axis=0)

    return point_coords, point_labels, points_fg, points_bg, box1, (cX, cY) 

def find_box_from_mask(mask):
    y, x = np.where(mask == 1)
    x0 = x.min()
    x1 = x.max()
    y0 = y.min()
    y1 = y.max()
    return [x0, y0, x1, y1]

    # get box and points information
def find_all_info1(mask, label_list):
    point_list = []
    point_label_list = []
    mask_list = []
    box_list = []
    # multi-object processing
    for current_label_id in range(len(label_list)):
        current_mask = mask[current_label_id]
        current_center_point_list, current_label_list,_,_,_,_=  find_center_from_mask_new(current_mask)
        current_box = find_box_from_mask(current_mask)
        point_list.append(current_center_point_list[0:10,:])
        point_label_list.append(current_label_list[0:10,])
        mask_list.append(current_mask)
        box_list.append(current_box)
    return point_list, point_label_list, box_list, mask_list

def read_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(mask_path,0)
    return image, gt

def get_one_target_final_dice(masks, gt):
    dice_list = []
    for i in range(len(masks)):
        dice_list.append(dice_coefficient(masks[i], gt))
    # choose the largest dice (GT <-> mask)
    res_mask = masks[np.argmax(dice_list)]
    return dice_list[np.argmax(dice_list)], res_mask

def dice_coefficient(y_true, y_pred):
    """
    y_true: GT, [N, W, H]
    Y_pred: target, [M, W, H]
    N, M: number
    W, H: weight and height of the masks
    Returns:
        dice_matrix [N, M]
        dice_max_index [N,] indexes of prediceted masks with the highest DICE between each N GTs 
        dice_max_value [N,] N values of the highest DICE
    """
    smooth = 0.1
    y_true_f = y_true.reshape(y_true.shape[0], -1) # N
    
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1) # M
    intersection = np.matmul(y_true_f.astype(float), y_pred_f.T.astype(float))
    dice_matrix = (2. * intersection + smooth) / (y_true_f.sum(1).reshape(y_true_f.shape[0],-1) + y_pred_f.sum(1) + smooth)
    dice_max_index, dice_max_value = dice_matrix.argmax(1), dice_matrix.max(1)
    return dice_matrix, dice_max_index, dice_max_value

def iou_coefficient(y_true, y_pred):
    """
    y_true: GT, [N, W, H]
    Y_pred: target, [M, W, H]
    N, M: number
    W, H: weight and height of the masks
    Returns:
        iou_matrix [N, M]
        iou_max_index [N,] indexes of predicted masks with the highest IoU between each N GTs 
        iou_max_value [N,] N values of the highest IoU
    """
    smooth = 0.1  # 用于避免除以零
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    intersection = np.matmul(y_true_f.astype(float), y_pred_f.T.astype(float))
    # 计算并集
    union = (y_true_f.sum(1).reshape(y_true_f.shape[0], -1) + y_pred_f.sum(1) - intersection)
    iou_matrix = (intersection) / (union)
    iou_max_index, iou_max_value = iou_matrix.argmax(1), iou_matrix.max(1)
    return iou_matrix, iou_max_index, iou_max_value

def f1_score_coefficient(y_true, y_pred):
    """
    y_true: GT, [N, W, H]
    y_pred: target, [M, W, H]
    N, M: number
    W, H: weight and height of the masks
    Returns:
        f1_matrix [N, M]
        f1_max_index [N,] indexes of predicted masks with the highest F1-score between each N GTs 
        f1_max_value [N,] N values of the highest F1-score
    """
    smooth = 0.1  # 用于避免除以零
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    # 计算intersection (TP)
    intersection = np.matmul(y_true_f.astype(float), y_pred_f.T.astype(float))
    # 计算precision和recall的分母
    y_true_sum = y_true_f.sum(1).reshape(y_true_f.shape[0], 1)  # GT中的正样本数 (TP + FN)
    y_pred_sum = y_pred_f.sum(1)  # 预测中的正样本数 (TP + FP)
    # 计算precision和recall
    # if y_pred_sum == 0:
    #     return 0, 0, 0
    precision = (intersection) / (y_pred_sum)  # TP / (TP + FP)
    recall = (intersection) / (y_true_sum)     # TP / (TP + FN)
    # 计算F1-score
    f1_matrix = 2 * (precision * recall) / (precision + recall)
    f1_max_index, f1_max_value = f1_matrix.argmax(1), f1_matrix.max(1)
    return f1_matrix, f1_max_index, f1_max_value

def oa_coefficient(y_true, y_pred):
    """
    y_true: GT, [N, W, H]
    y_pred: target, [M, W, H]
    N, M: number
    W, H: weight and height of the masks
    Returns:
        oa_matrix [N, M] - pixel-wise accuracy matrix between each GT and predicted mask
        oa_max_index [N,] - indexes of predicted masks with the highest OA between each N GTs 
        oa_max_value [N,] - N values of the highest OA
    """
    smooth = 0.1  # 用于避免除以零
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    # 计算每对mask之间的像素级准确率
    # 对于二值mask，准确率 = (TP + TN) / (TP + TN + FP + FN)
    total_pixels = y_true_f.shape[1]  # W * H
    # 计算正确预测的像素数：相同位置都是1或都是0的像素
    # 使用矩阵运算批量计算所有GT和预测mask对之间的准确率
    oa_matrix = np.zeros((y_true_f.shape[0], y_pred_f.shape[0]))
    for i in range(y_true_f.shape[0]):
        for j in range(y_pred_f.shape[0]):
            # 计算正确预测的像素数 (TP + TN)
            correct_pixels = np.sum(y_true_f[i] == y_pred_f[j])
            oa_matrix[i, j] = (correct_pixels) / (total_pixels)
    oa_max_index, oa_max_value = oa_matrix.argmax(1), oa_matrix.max(1)
    return oa_matrix, oa_max_index, oa_max_value

if __name__ == "__main__":
    # task_name
    task = "Vihaigen"
    print("Current processing task: ", task)
    mode = "test"
    # model size
    size = "b" 
        
    sam_checkpoint = f"work_dir_b/finetune_data/medsam_box_best.pth"
    json_info_path = f"data_infor_json/{task}.json"
    test_mode = "finetune_data"
    model_type = f"vit_{size}"
    device = "cuda:0"
        
    """construct model and predictor"""
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, device=device)
    sam.to(device)
    sam_transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.eval()
    predictor = SamPredictor(sam)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    #acquire dataset infomarion
    info = json.load(open(json_info_path))
    label_info = info["info"]
    color_info = info["color"]

    # dice
    all_dices = {}
    all_iou = {}
    all_f1 = {}
    all_oa = {}
    # save infer results (png: final mask, npy: pred_all_masks)
    # save selected prompts (npz)
    save_base = f"work_dir_{size}/{test_mode}/{task}/pred_only_box"
    save_prompt = f"work_dir_{size}/{test_mode}/{task}/prompts_only_box"
    os.makedirs(save_base, exist_ok=True)
    os.makedirs(save_prompt, exist_ok=True)
    
    all_dices["box"] = {}
    all_iou["box"] = {}
    all_f1["box"] = {}
    all_oa["box"] = {}
    
    dice_targets = [[] for _ in range(len(label_info))] # 某个方法中,记录不同结构的dice
    iou_targets = [[] for _ in range(len(label_info))]
    f1_targets = [[] for _ in range(len(label_info))]
    oa_targets = [[] for _ in range(len(label_info))]
    with tqdm(total=len(glob.glob(f"ISPRS_dataset/valid_data_potsdam/images/*")), desc=f'Current mode: box', mininterval=0.3) as pbar:
        for ori_path in glob.glob(f"ISPRS_dataset/valid_data_potsdam/images/*"):
            origin_name = ori_path
            name = ori_path.split('/')[-1]
            # print(origin_name)
            gt_data = io.imread(join("ISPRS_dataset/valid_data_potsdam/labels/", name))
            
#             if os.path.exists(join(f"ISPRS_dataset/Vaihingen/precompute_vit_{size}/test", task, name + ".npz")):
#                 npz_data = np.load(join(f"ISPRS_dataset/Vaihingen/precompute_vit_{size}/test", task, name + ".npz"))
#             else:
#                 continue
#             gt2D = npz_data['gts']
#             label_list = npz_data['new_lab_list']
#             img_embed = torch.tensor(npz_data['img_embeddings']).float()
#             dsm_embed = torch.tensor(npz_data['dsm_embedding']).float()
#             # prompt mode
#             predictor.original_size = tuple(npz_data['image_shape'])
#             predictor.input_size = tuple(npz_data['resized_size_before_padding'])
#             predictor.features = (img_embed.unsqueeze(0).to(device), dsm_embed.unsqueeze(0).to(device))
            
#             if not os.path.exists(os.path.join(save_base, ori_path.split('/')[-1].replace('png', 'npy'))):
#                 if not os.path.exists(f"{save_prompt}/{name}_prompts.npz"):#如果不存在就创建一个prompts.npz文件
#                     _, _, box_list, gt_list = find_all_info(gt2D, label_list)
#                     box_list = np.array(box_list)
#                     np.savez(f"{save_prompt}/{name}_prompts.npz", box_list, gt_list)
#                 else:
#                     info = np.load(f"{save_prompt}/{name}_prompts.npz")  
#                     box_list, gt_list = info['arr_0'], info['arr_1']
                    
#                 # pre_process
#                 # box_list_tensor = torch.tensor(box_list).float().to(device)
#                 box_list_tensor = torch.tensor(npz_data['label_except_bk'])
#                 point_label_list = torch.tensor(npz_data['point_label_list'])
#                 point = torch.tensor(npz_data['point'])
            label_p = gt_data
            gt = label_p.copy()
            del_lab = []
            if np.unique(label_p)[0] != 0:
                label_list = np.unique(label_p)
                for label in label_list:
                    gt_single = (label_p == label) + 0
                    if np.sum(gt_single) < 1:
                        gt[gt == label] = 0
                        del_lab.append(label)
                        assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) ) , \
                        f"标签数量不匹配！处理后标签: {np.unique(gt)}, 删除标签: {del_lab}, 原始标签: {np.unique(label_p)}"
            else:
                label_list = np.unique(label_p)[0:]
                for label in label_list:
                    gt_single = (label_p == label) + 0
                    if np.sum(gt_single) < 1:
                        gt[gt == label] = 0
                        del_lab.append(label)
                assert len(list(np.unique(gt)) + del_lab) == len(list(label_list)), \
                f"标签数量不匹配！处理后标签: {np.unique(gt)}, 删除标签: {del_lab}, 原始标签: {np.unique(label_p)}"

                # for label in label_list:
                #     gt_single = (label_p == label) + 0
                #     if np.sum(gt_single) <= 50:
                #         gt[gt == label] = 0
                #         del_lab.append(label)

                # assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0])

            new_lab_list = list(np.unique(gt)) # except bk
            
            new_lab_list.sort()
            lab_list=new_lab_list
                # gt_ = []
                # for l in new_lab_list:
                #     gt_.append((gt == l) + 0)
            if not new_lab_list:  # 处理全背景情况
                    # 方法：生成一个空的全零掩码（通道数为0）或强制1个通道
                    # 这里选择强制1个通道（全零），避免维度错误
                h, w = gt.shape[:2]
                gt_ = np.zeros((1, h, w), dtype=np.uint8)
            else:
                gt_ = []
                for l in new_lab_list:
                    gt_.append((gt == l).astype(np.uint8))
            gt_ = np.array(gt_, dtype=np.uint8)
            
            if not new_lab_list:
                box_list = np.zeros((1, 4), dtype=np.uint8)
                point_list = np.zeros((1, 10, 2), dtype=np.uint8)
                point_label_list = np.zeros((1, 10), dtype=np.uint8)
            else:
                # box_list = np.zeros((gt_.shape[0], 4), dtype=np.uint8)
                point_list, point_label_list, box_list = find_all_info(mask=gt_, 
                                                label_list=new_lab_list, 
                                                point_num=150)
                    # print(point_list.shape)
                    # print(point_label_list)
                    # convert img embedding, gt_mask, bounding box to torch tensor
            box = torch.tensor(box_list).float() # B, 4
            box = resize_transform.apply_boxes_torch(box, (gt_.shape[-2], gt_.shape[-1]))
            
            point = torch.tensor(point_list).float() # B, 4
            point_label_list = torch.tensor(point_label_list).float()
            point = resize_transform.apply_coords_torch(point, (gt_.shape[-2], gt_.shape[-1]))
            if np.sum(gt) >= 0: # has at least 1 object
                # gt: seperate each target into size (B, H, W) binary 0-1 uint8
                # new_lab_list = list(np.unique(gt))[1:] # except bk
                # new_lab_list.sort()
                gt_ = []
                for l in new_lab_list:
                    gt_.append((gt == l) + 0)
                gt_ = np.array(gt_, dtype=np.uint8)
                gt_list = gt_
                if mode == "train":
                    image_data = io.imread(join(args.img_path, image_name))
                elif mode == "valid":
                    image_data = io.imread(join(args.img_path.replace("train", "valid"), image_name))
                else:
                    
                    img_path = "ISPRS_dataset/valid_data_potsdam/images"
                    dsm_path = "ISPRS_dataset/valid_data_potsdam/dsm"
                    data_p = io.imread(join(img_path, name), dtype='float32')
                    dsm_p = np.asarray(io.imread(join(dsm_path, name), dtype='float32'))
                    
                    min = np.min(dsm_p)
                    max = np.max(dsm_p)
                    
                    smooth = 0
                    if max == 0:
                        smooth = 1
                    dsm_p = (dsm_p - min) / (max - min + smooth)
                    
                    data_p = torch.tensor(data_p).to(device)
                    dsm_p = torch.tensor(dsm_p).to(device)
                    data_p = data_p.float()/255.0
                    dsm_p = dsm_p.float()
                    
                image_ori_size = data_p.shape[:2]
                data_p = upsample_patch(data_p.permute(2, 0, 1))
                dsm_p = upsample_patch(dsm_p)

        #         # Remove any alpha channel if present.
        #         if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
        #             image_data = image_data[:, :, :3]
        #         # If image is grayscale, then repeat the last channel to convert to rgb
        #         if len(image_data.shape) == 2:
        #             image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        #         # nii preprocess start (clip the intensity)
        #         lower_bound, upper_bound = np.percentile(image_data, 0.95), np.percentile(
        #             image_data, 99.5 # Intensity of 0.95% pixels in image_data lower than lower_bound
        #                              # Intensity of 99.5% pixels in image_data lower than upper_bound
        #         )
        #         image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        #         # min-max normalize and scale
        #         image_data_pre = (
        #             (image_data_pre - np.min(image_data_pre))
        #             / (np.max(image_data_pre) - np.min(image_data_pre))
        #             * 255.0
        #         )
        #         image_data_pre[image_data == 0] = 0 # ensure 0-255
        #         image_data_pre = np.uint8(image_data_pre)
        #         imgs.append(image_data_pre)

        #         # resize image to 3*1024*1024
        #         resize_img = sam_transform.apply_image(image_data_pre, interpolation=InterpolationMode.BILINEAR) # ResizeLong
        #         resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1))[None, :, :, :].to(
        #             args.device
        #         ) # (1, 3, resized_h, 1024)
        #         resized_size_be_padding = tuple(resize_img_tensor.shape[-2:])
        #         input_image = sam_model.preprocess(resize_img_tensor) # padding to (1, 3, 1024, 1024)
        #         assert input_image.shape == (
        #             1,
        #             3,
        #             sam_model.image_encoder.img_size,
        #             sam_model.image_encoder.img_size,
        #         ), "input image should be resized to 1024*1024"
        #         assert input_image.shape[-2:] == (1024, 1024)
        #         # pre-compute the image embedding
                if mode != "train":
                    sam.eval()
                with torch.no_grad():
                    img_embedding, dsm_embedding = sam.image_encoder(data_p.unsqueeze(0).to(device), dsm_p.unsqueeze(0).to(device))
                    # img_embedding, dsm_embedding = embedding.cpu().numpy()[0], dsm.cpu().numpy()[0]
                resized_size_be_padding = (1024, 1024)
                predictor.original_size = tuple(image_ori_size)
                predictor.input_size = tuple(resized_size_be_padding)
                predictor.features = (img_embedding.to(device), dsm_embedding.to(device))
                '''box'''
                masks, scores, logits = predictor.predict_torch(
                    point_coords=point.to(device),
                    point_labels=point_label_list.to(device),
                    boxes = box.to(device),
                    multimask_output=True,
                    ) # Mask -> N,M,H,W
                
                masks = masks.cpu().numpy()  
                np.save(os.path.join(save_base, ori_path.split('/')[-1].replace('png', 'npy')), masks)
                
            else:
                gt_list = np.load(f"{save_prompt}/{name}_prompts.npz")['arr_1']
                masks = np.load(os.path.join(save_base, ori_path.split('/')[-1].replace('png', 'npy')), allow_pickle=True)[()]
            
            # compute dice 
            current_method_res = np.zeros((256, 256)) # all target in a single image
            for idx in range(len(gt_list)): # mask list for a single image
                current_gt = gt_list[idx]
                dice_matrix, dice_max_index, dice_max_value = dice_coefficient(y_true=current_gt[None, :, :], y_pred=masks[idx])
                _, _, iou_max_value = iou_coefficient(y_true=current_gt[None, :, :], y_pred=masks[idx])
                _, _, f1_max_value = f1_score_coefficient(y_true=current_gt[None, :, :], y_pred=masks[idx])
                _, _, oa_max_value = oa_coefficient(y_true=current_gt[None, :, :], y_pred=masks[idx])
                # print(dice_matrix)
                # print(dice_max_index)
                # print(dice_max_value)
                final_mask = masks[idx][dice_max_index.squeeze(0)]
                # print(list(color_info.values()))
                # print(label_list[idx])
                # id_dice = int(list(color_info.keys())[list(color_info.values()).index(label_list[idx])])
                # dice_targets[id_dice - 1].append(dice_max_value.squeeze(0))
                # current_method_res[final_mask == 1] = label_list[idx] # one target
                # index mapping for matching DICE with different target structures
                try:
                    id_dice = int(list(color_info.keys())[list(color_info.values()).index(label_list[idx])])
                    dice_targets[id_dice].append(dice_max_value.squeeze(0))
                    current_method_res[final_mask == 1] = label_list[idx] # one target
                    
                    iou_targets[id_dice - 1].append(iou_max_value.squeeze(0))
                    f1_targets[id_dice - 1].append(f1_max_value.squeeze(0))
                    oa_targets[id_dice - 1].append(oa_max_value.squeeze(0))
                except:
                    print(f"error: {ori_path}")
                    continue
            
            # for visulize (save infer results)
            if not os.path.exists(os.path.join(save_base, ori_path.split('/')[-1])):
                cv2.imwrite(os.path.join(save_base, ori_path.split('/')[-1]), current_method_res.astype(np.uint8))
                    
            pbar.update(1)
            
        # print
        for id in range(len(dice_targets)):
            all_dices["box"][label_info[str(id)]] = f'{round(np.array(dice_targets[id]).mean() * 100, 2)}({round((100 * np.array(dice_targets[id])).std(), 2)})'
        for id in range(len(dice_targets)):
            all_iou["box"][label_info[str(id)]] = f'{round(np.array(iou_targets[id]).mean() * 100, 2)}({round((100 * np.array(iou_targets[id])).std(), 2)})'
        for id in range(len(dice_targets)):
            all_f1["box"][label_info[str(id)]] = f'{round(np.array(f1_targets[id]).mean() * 100, 2)}({round((100 * np.array(f1_targets[id])).std(), 2)})'
        for id in range(len(dice_targets)):
            all_oa["box"][label_info[str(id)]] = f'{round(np.array(oa_targets[id]).mean() * 100, 2)}({round((100 * np.array(oa_targets[id])).std(), 2)})'
        print("======following is the dice results======")
        print(json.dumps(all_dices, indent=4, ensure_ascii=False))
        print(json.dumps(all_iou, indent=4, ensure_ascii=False))
        print(json.dumps(all_f1, indent=4, ensure_ascii=False))
        print(json.dumps(all_oa, indent=4, ensure_ascii=False))


        
