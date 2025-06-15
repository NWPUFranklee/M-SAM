# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
import random
join = os.path.join
from tqdm import tqdm
from time import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.get_prompts import find_all_info
import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from utils1 import *

# set seeds
torch.manual_seed(3407)
np.random.seed(2023)

from torch.autograd import Variable
def collate_fn(batch):
    return batch

#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.resize_transform = ResizeLongestSide(target_length=1024)
    
    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        # print(self.npz_files[index])
        npz_data = np.load(join(self.data_root, self.npz_files[index]))
        gt2D = npz_data['gts']
        label_list = npz_data['label_except_bk'].tolist()
        img_embed = npz_data['img_embeddings']
        dsm_embed = npz_data['img_embeddings_dsm']
        _, _, box_list = find_all_info(mask=gt2D, 
                                       label_list=label_list, 
                                       point_num=4
                                       )
        # convert img embedding, gt_mask, bounding box to torch tensor
        box = torch.tensor(box_list).float() # B, 4
        img_embed = torch.tensor(img_embed).float()
        gt2D = torch.tensor(gt2D).long() # B, ori_H, ori_W

        # scale to original size
 
        box = self.resize_transform.apply_boxes_torch(box, (gt2D.shape[-2], gt2D.shape[-1]))
        
        return {"img_embed": img_embed,
                "gt2D": gt2D,
                "box": box,
                "image_ori_size": npz_data['image_shape'],
                "size_before_pad": npz_data['resized_size_before_padding']
                }
        
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
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
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

    iou_matrix = (intersection + smooth) / (union + smooth)
    
    iou_max_index, iou_max_value = iou_matrix.argmax(1), iou_matrix.max(1)
    return iou_matrix, iou_max_index, iou_max_value

def get_iou(y_true, y_pred):
    """
    计算二分类（0和1）掩码的交并比(IOU)
    
    Parameters
    ----------
    y_true : numpy.ndarray
        真实掩码，形状为 [N, W, H]，N为掩码数量，值为0或1
    y_pred : numpy.ndarray
        预测掩码，形状为 [M, W, H]，M为掩码数量，值为0或1
    
    Returns
    -------
    iou_matrix : numpy.ndarray
        IOU矩阵，形状为 [N, M]，表示每个真实掩码与每个预测掩码的IOU
    iou_max_index : numpy.ndarray
        每个真实掩码对应的最佳预测掩码索引，形状为 [N, ]
    iou_max_value : numpy.ndarray
        每个真实掩码对应的最大IOU值，形状为 [N, ]
    """
    # 确保输入为numpy数组并转换为布尔类型
    y_true = np.array(y_true).astype(bool)
    y_pred = np.array(y_pred).astype(bool)
    
    # 展平为二维矩阵 [N, W*H] 和 [M, W*H]
    true_flat = y_true.reshape(y_true.shape[0], -1)
    pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    
    # 计算交集：[N, M]
    intersection = np.matmul(true_flat, pred_flat.T)
    
    # 计算真实掩码和预测掩码的前景像素数
    true_sum = true_flat.sum(axis=1, keepdims=True)  # [N, 1]
    pred_sum = pred_flat.sum(axis=1)                 # [M, ]
    
    # 计算并集：[N, M]
    union = true_sum + pred_sum - intersection
    
    # 计算IOU，处理除零情况
    iou_matrix = np.where(union == 0, 0.0, intersection / union)
    
    # 获取每个真实掩码对应的最佳预测掩码
    iou_max_index = iou_matrix.argmax(1)
    iou_max_value = iou_matrix.max(1)
    
    return iou_matrix, iou_max_index, iou_max_value

def get_val_dataset(model: nn.Module, epoch, batch_size):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=Stride_Size, window_size=WINDOW_SIZE) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=Stride_Size, window_size=WINDOW_SIZE)), total=total,
                    leave=False)):
                # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
            dsm_patches = np.asarray(dsm_patches)
            dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)
                
            label_patches = [np.copy(gt_e[x:x + w, y:y + h]) for x, y, w, h in coords]
            label_patches = np.asarray(label_patches)
            label_p = Variable(torch.from_numpy(label_patches).cuda(), volatile=True)

            
            for i in range(batch_size):
                data_p = model.preprocess(torch.tensor(image_patches[0]))
                dsm_p = model.preprocess(torch.tensor(dsm_patches[0]))
            gt = label_p.copy()
            del_lab = []
            if np.unique(label_p)[0] != 0:
                label_list = np.unique(label_p)
                for label in label_list:
                    gt_single = (label_p == label) + 0
                    if np.sum(gt_single) <= 50:
                        gt[gt == label] = 0
                        del_lab.append(label)
                        assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0]) , \
                        f"标签数量不匹配！处理后标签: {np.unique(gt)}, 删除标签: {del_lab}, 原始标签: {np.unique(label_p)}"
            else:
                label_list = np.unique(label_p)[1:]
                for label in label_list:
                    gt_single = (label_p == label) + 0
                    if np.sum(gt_single) <= 50:
                        gt[gt == label] = 0
                        del_lab.append(label)
                assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0]), \
                    f"标签数量不匹配！处理后标签: {np.unique(gt)}, 删除标签: {del_lab}, 原始标签: {np.unique(label_p)}"

                # for label in label_list:
                #     gt_single = (label_p == label) + 0
                #     if np.sum(gt_single) <= 50:
                #         gt[gt == label] = 0
                #         del_lab.append(label)

                # assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0])

            new_lab_list = list(np.unique(gt))[1:] # except bk
            new_lab_list.sort()
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
            else:
                 _, _, box_list = find_all_info(mask=gt_, 
                                                label_list=new_lab_list, 
                                                point_num=10)
                # convert img embedding, gt_mask, bounding box to torch tensor
            box = torch.tensor(box_list).float() # B, 4
            box = self.resize_transform.apply_boxes_torch(box, (gt_.shape[-2], gt_.shape[-1]))

            image_record = {"img_embed": data_p,
                        "dsm_embed":dsm_p,
                        "gt2D": torch.tensor(gt_),
                        "box": box,
                        "image_ori_size": [256, 256],
                        "size_before_pad": [1024, 1024]
            }
            image_record["img_embed"] = model.image_encoder(image_record["img_embed"].unsqueeze(0).to(device)).squeeze(0)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=image_record["box"].to(device),
                masks=None,
            )
                # low_res_masks.shape == (B, M, 256, 256) M is set to 1
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_record["img_embed"].unsqueeze(0).to(device), # (1, 256, 64, 64) !!1 = batch size
                image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64) !!1 = batch size
                sparse_prompt_embeddings=sparse_embeddings, # (B, N, 256) !!B = target num instead of batch size
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64) !!B = target num instead of batch size
                multimask_output=True,
            )
                
            masks = model.postprocess_masks( # upscale + eliminate padding + restore to ori size
                low_res_masks,
                input_size=tuple(image_record["size_before_pad"]),
                original_size=tuple(image_record["image_ori_size"]),
            )
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
                "gt2D": image_record["gt2D"].to(device)
            })
                # compute foreground dice
            masks = masks > model.mask_threshold
            masks = masks.cpu().numpy() # B, 1, ori_H, ori_W
            gt2D = np.array(image_record["gt2D"]) # B, ori_H, ori_W
            target_mean_dsc = 0.
            for i in range(len(gt2D)): # each target
                cur_gt = gt2D[i]
                _, _, dice_max_value = dice_coefficient(y_true=cur_gt[None, :, :], y_pred=masks[i])
                target_mean_dsc += dice_max_value.squeeze(0)
                batched_dsc += target_mean_dsc / len(gt2D)
                
                _, _, iou_max_value = iou(y_true=cur_gt[None, :, :], y_pred=masks[i])
                target_mean_iou += iou_max_value.squeeze(0)
                batched_iou += target_mean_iou / len(gt2D)                
                
        dsc = batched_dsc / len(i)
        epoch_dsc += dsc
        
        iou = batched_iou / len(i)
        epoch_iou += iou
        
    epoch_dsc /= (len(test_ids))
    epoch_iou /= (len(test_ids))
    print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, DSC: {epoch_dsc}, IOU: {epoch_iou}')
    return epoch_dsc
def train_sam(model: nn.Module, optimizer, train_loader, epoch, device, criterion):
    epoch_start_time = time()
    epoch_loss = 0
    model.train()
    print("==========Training==========")
    for step, batched_input in enumerate(tqdm(train_loader)):
        outputs = []
        # do not compute gradients for image encoder and prompt encoder
        # with torch.no_grad():
        none_grad_features = {"sparse": {}, "dense": {}}
        for idx, image_record in enumerate(batched_input):
            # print(image_record["img_embed"].shape)
            img_encoder =  model.image_encoder
            
            image_record["img_embed"], image_record["dsm_embed"] = img_encoder(image_record["img_embed"].unsqueeze(0).to(device), image_record["dsm_embed"].unsqueeze(0).to(device))
            image_record["img_embed"], image_record["dsm_embed"] = image_record["img_embed"].squeeze(0), image_record["dsm_embed"].squeeze(0)
            # image_record["dsm_embed"] = img_encoder(image_record["dsm_embed"].unsqueeze(0).to(device)).squeeze(0)
            # image_record["img_embed"] = img_encoder(image_record["img_embed"].unsqueeze(0).to(device)).squeeze(0)
            # image_record["dsm_embed"] = model.image_encoder(image_record["dsm_embed"].unsqueeze(0).to(device))
            # print(image_record["box"])
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(image_record["point"].to(device), image_record["point_label"].to(device)),
                boxes=image_record["box"].to(device),
                masks=None,
            )
            none_grad_features["sparse"][idx] = sparse_embeddings
            none_grad_features["dense"][idx] = dense_embeddings

            # print(image_record["gt2D"].shape)
        batched_loss = 0
        for id, im_record in enumerate(batched_input):
            # low_res_masks.shape == (B, M, 256, 256) M is set to 1
            # print("******")
            # print(im_record["img_embed"].unsqueeze(0).shape)
            # print(model.prompt_encoder.get_dense_pe().shape)
            # print(none_grad_features["sparse"][id].shape)
            # print(none_grad_features["dense"][id].shape)
            # print("******")
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=(im_record["img_embed"].unsqueeze(0).to(device), im_record["dsm_embed"].unsqueeze(0).to(device)), # (1, 256, 64, 64) !!1 = batch size
                image_pe=(model.prompt_encoder.get_dense_pe(),model.prompt_encoder.get_dense_pe(),model.prompt_encoder.get_dense_pe()), # (1, 256, 64, 64) !!1 = batch size
                sparse_prompt_embeddings=none_grad_features["sparse"][id], # (B, 2, 256) !!B = target num instead of batch size
                dense_prompt_embeddings=none_grad_features["dense"][id], # (B, 256, 64, 64) !!B = target num instead of batch size
                multimask_output=False,
            )
            # print(im_record["gt2D"].shape)
            # print(im_record["gt2D"].shape)
            # print(im_record["size_before_pad"])
            # print(im_record["image_ori_size"])
            masks = model.postprocess_masks( # upscale + eliminate padding + restore to ori size
                low_res_masks,
                input_size=tuple(im_record["size_before_pad"]),
                original_size=tuple(im_record["image_ori_size"]),
            )
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
                "gt2D": im_record["gt2D"].to(device)
            })
            # first ele: 1, B, ori_H, ori_W
            # second ele: 1, B, ori_H, ori_W
            batched_loss += criterion(masks.squeeze(1).unsqueeze(0), im_record["gt2D"].to(device).unsqueeze(0)) # considering the multi-object situation
        loss = batched_loss / len(batched_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= (step + 1)
    print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
    return epoch_loss, time() - epoch_start_time

def valid_sam(model: nn.Module, valid_loader, epoch, device):
    model.eval()
    epoch_dsc = 0.
    epoch_iou = 0.
    print("==========Validation==========")
    for step, batched_input in enumerate(tqdm(valid_loader)):
        outputs = []
        with torch.no_grad():
            batched_dsc = 0.
            batched_iou = 0.
            for image_record in batched_input:
                # image_record["dsm_embed"] = model.image_encoder(image_record["dsm_embed"].unsqueeze(0).to(device)).squeeze(0)
                # image_record["img_embed"] = model.image_encoder(image_record["img_embed"].unsqueeze(0).to(device)).squeeze(0)
                img_encoder =  model.image_encoder
                # print(image_record["dsm_embed"].shape)
                # print(image_record["img_embed"].shape)
                image_record["img_embed"], image_record["dsm_embed"] = img_encoder(image_record["img_embed"].unsqueeze(0).to(device), image_record["dsm_embed"].unsqueeze(0).to(device))
                image_record["img_embed"], image_record["dsm_embed"] = image_record["img_embed"].squeeze(0), image_record["dsm_embed"].squeeze(0)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(image_record["point"].to(device), image_record["point_label"].to(device)),
                    boxes=image_record["box"].to(device),
                    masks=None,
                )
                # low_res_masks.shape == (B, M, 256, 256) M is set to 1
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=(image_record["img_embed"].unsqueeze(0).to(device), image_record["dsm_embed"].unsqueeze(0).to(device)), # (1, 256, 64, 64) !!1 = batch size
                    image_pe=(model.prompt_encoder.get_dense_pe(),model.prompt_encoder.get_dense_pe(),model.prompt_encoder.get_dense_pe()), # (1, 256, 64, 64) !!1 = batch size
                    sparse_prompt_embeddings=sparse_embeddings, # (B, N, 256) !!B = target num instead of batch size
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64) !!B = target num instead of batch size
                    multimask_output=True,
                )
                
                masks = model.postprocess_masks( # upscale + eliminate padding + restore to ori size
                    low_res_masks,
                    input_size=tuple(image_record["size_before_pad"]),
                    original_size=tuple(image_record["image_ori_size"]),
                )
                
                outputs.append({
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "gt2D": image_record["gt2D"].to(device)
                })
                # compute foreground dice
                masks = masks > model.mask_threshold
                masks = masks.cpu().numpy() # B, 1, ori_H, ori_W
                gt2D = np.array(image_record["gt2D"]) # B, ori_H, ori_W
                target_mean_dsc = 0.
                target_mean_iou = 0.
                for i in range(len(gt2D)): # each target
                    cur_gt = gt2D[i]
                    _, _, dice_max_value = dice_coefficient(y_true=cur_gt[None, :, :], y_pred=masks[i])
                    target_mean_dsc += dice_max_value.squeeze(0)
                    
                    _, _, iou_max_value = iou_coefficient(y_true=cur_gt[None, :, :], y_pred=masks[i])
                    target_mean_iou += iou_max_value.squeeze(0)
                batched_iou += target_mean_iou / len(gt2D)                
                batched_dsc += target_mean_dsc / len(gt2D)
        dsc = batched_dsc / len(batched_input)
        epoch_dsc += dsc
        
        iou = batched_iou / len(batched_input)
        epoch_iou += iou
    epoch_dsc /= (step + 1)
    epoch_iou /= (step + 1)
    print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, DSC: {epoch_dsc}, IOU: {epoch_iou}')
    return epoch_dsc

if __name__ == "__main__":
    # Task = "Heart_2d"
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_npz_path', type=str, default=f'data/precompute_vit_b/train', help='path to training npz files (im_emb, gt)')
    parser.add_argument('-j', '--val_npz_path', type=str, default=f'data/precompute_vit_b/train', help='path to validation npz files (im_emb, gt)')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, default='sam_vit_b_01ec64.pth', help='original sam checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--work_dir', type=str, default='work_dir_b')
    # train
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()


    # %% set up model for fine-tuning 
    Task = "finetune_data"
    device = args.device
    model_save_path = join(args.work_dir, Task)
    os.makedirs(model_save_path, exist_ok=True)
    # print(args.model_type)
    # print(args.checkpoint)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, device=device).to(device)

    # Set up the optimizer, hyperparameter tuning will improve performance here
    optimizer = torch.optim.AdamW(sam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    #%% train & valid
    num_epochs = args.num_epochs
    losses = []
    dscs = []
    times = []
    best_loss, best_dsc = 1e10, 0.
    train_dataset = ISPRS_dataset(train_ids, "train")
    val_dataset = ISPRS_dataset(test_ids, "test")
    print("Number of training samples: ", len(train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                  pin_memory=True, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                  pin_memory=True, shuffle=False, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        # train
        epoch_loss, runtime = train_sam(sam_model, optimizer, train_dataloader, epoch, device, criterion=seg_loss)
        losses.append(epoch_loss)
        times.append(runtime)
        # valid
        epoch_dsc = valid_sam(sam_model, valid_dataloader, epoch, device)
        # epoch_dsc = get_val_dataset(sam_model,  epoch, args.batch_size)
        dscs.append(epoch_dsc)
        
        # save the model checkpoint
        torch.save(sam_model.state_dict(), join(model_save_path, f'medsam_box_last.pth'))
        # save the best model
        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            print("Update: saving {} model as the best checkpoint".format(epoch))
            torch.save(sam_model.state_dict(), join(model_save_path, f'medsam_box_best.pth'))

        # %% plot loss
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
        ax1.plot(losses)
        ax1.set_title("Dice + Cross Entropy Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.plot(dscs)
        ax2.set_title("Dice of valid set")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice")
        ax3.plot(times)
        ax3.set_title("Epoch Running Time")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Time (s)")
        fig.savefig(join(model_save_path, f"medsam_box_record.png"))
        plt.close()

