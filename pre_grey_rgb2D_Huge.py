# %% import packages
import numpy as np
import os
from glob import glob
import pandas as pd
import torch.nn.functional as F
join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from torchvision.transforms.functional import InterpolationMode
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from utils.get_prompts import find_all_info
resize_transform = ResizeLongestSide(target_length=1024)

# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")
# add arguments to the parser
parser.add_argument(
    "-i",
    "--img_path",
    type=str,
    default=f"data_vaihin_cropped_256/test_data/",
    help="path to the images",
)
parser.add_argument(
    "-d",
    "--dsm_path",
    type=str,
    default=f"data_vaihin_cropped_256/test_data/",
    help="path to the dsm",
)
parser.add_argument(
    "-gt",
    "--gt_path",
    type=str,
    default=f"data_vaihin_cropped_256/test_data/",
    help="path to the ground truth (gt)",
)

parser.add_argument(
    "-task",
    "--task_name",
    type=str,
    default=f"Vihaigen",
    help="name to test dataset",
)

parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="path to the csv file",
)

parser.add_argument(
    "-o",
    "--npz_path",
    type=str,
    default=f"ISPRS_dataset/Vaihingen",
    help="path to save the npz files",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="demo2d",
    help="dataset name; used to name the final npz file, e.g., demo2d.npz",
)
parser.add_argument("--image_size", type=int, default=1024, help="image size")
parser.add_argument(
    "--img_name_suffix", type=str, default=".tif", help="image name suffix"
)
# parser.add_argument("--label_id", type=int, default=255, help="label id")
parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="work_dir_b/Potsdam/medsam_box_best.pth",
    help="checkpoint",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()

# create a directory to save the npz files
# save_base = args.npz_path + "/precompute_" + args.model_type
save_base = args.npz_path + "/precompute_" + args.model_type
# convert 2d grey or rgb images to npz file
imgs = []
gts = []
img_embeddings = []

# set up the model
# get the model from sam_model_registry using the model_type argument
# and load it with checkpoint argument
# download save the SAM checkpoint.
# [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](VIT-B SAM model)

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, device = args.device).to(
    args.device
)

# ResizeLongestSide (1024), including image and gt
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
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
    
    label_p = gt_data
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
        point_list = np.zeros((1, 10, 2), dtype=np.uint8)
        point_label_list = np.zeros((1, 10), dtype=np.uint8)
    else:
        # box_list = np.zeros((gt_.shape[0], 4), dtype=np.uint8)
        point_list, point_label_list, box_list = find_all_info(mask=gt_, 
                                        label_list=new_lab_list, 
                                        point_num=50)
            # print(point_list.shape)
            # print(point_label_list)
            # convert img embedding, gt_mask, bounding box to torch tensor
    box = torch.tensor(box_list).float() # B, 4
    box = resize_transform.apply_boxes_torch(box, (gt_.shape[-2], gt_.shape[-1]))
        
    point = torch.tensor(point_list).float() # B, 4
    point_label_list = torch.tensor(point_label_list).float()
    print(point.shape)
    print(point_label_list.shape)
    point = resize_transform.apply_coords_torch(point, (gt_.shape[-2], gt_.shape[-1]))
    print(point.shape)
    if np.sum(gt) > 0: # has at least 1 object
        # gt: seperate each target into size (B, H, W) binary 0-1 uint8
        # new_lab_list = list(np.unique(gt))[1:] # except bk
        # new_lab_list.sort()
        gt_ = []
        for l in new_lab_list:
            gt_.append((gt == l) + 0)
        gt_ = np.array(gt_, dtype=np.uint8)

        if mode == "train":
            image_data = io.imread(join(args.img_path, image_name))
        elif mode == "valid":
            image_data = io.imread(join(args.img_path.replace("train", "valid"), image_name))
        else:
            img_path = f"{args.gt_path}/images"
            dsm_path = f"{args.gt_path}/dsm"
            data_p = io.imread(join(img_path, image_name))
            dsm_p = io.imread(join(dsm_path, image_name))
            data_p = torch.tensor(data_p).to(args.device)
            dsm_p = torch.tensor(dsm_p).to(args.device)
            data_p = data_p.float() / 255.0
            dsm_p = dsm_p.float() / 255.0
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
            sam_model.eval()
        with torch.no_grad():
            embedding, dsm = sam_model.image_encoder(data_p.unsqueeze(0).to(args.device), dsm_p.unsqueeze(0).to(args.device))
            img_embedding, dsm_embedding = embedding.cpu().numpy()[0], dsm.cpu().numpy()[0]
        resized_size_be_padding = (1024, 1024)
        return gt_, box, point, point_label_list,img_embedding, dsm_embedding, resized_size_be_padding, image_ori_size, new_lab_list
    else:
        print(mode, gt_name)
        return None, None, None, None, None

if __name__ == "__main__":
    mode = 'test'
    if args.csv != None:
        # if data is presented in csv format
        # columns must be named image_filename and mask_filename respectively
        try:
            os.path.exists(args.csv)
        except FileNotFoundError as e:
            print(f"File {args.csv} not found!!")

        df = pd.read_csv(args.csv)
        bar = tqdm(df.iterrows(), total=len(df))
        for idx, row in bar:
            process(row.mask_filename, row.image_filename)
    else:
         # get all the names of the images in the ground truth folder
        if mode == 'train' or mode == 'valid':
            names = sorted(os.listdir(args.gt_path))
            # save
            save_path = join(save_base, mode)
        else:
            gt_path = f"{args.gt_path}/labels"
            args.img_path = f"{args.gt_path}/images"
            args.dsm_path = f"{args.gt_path}/images"
            names = sorted(os.listdir(gt_path))
            # save
            save_path = join(save_base, mode,args.task_name)
        # print the number of images found in the ground truth folder
        print("Num. of all train images:", len(names))
        
        os.makedirs(save_path, exist_ok=True)
        for gt_name in tqdm(names):
            if os.path.exists(join(save_path, gt_name.split('.')[0] + ".npz")):
                continue
            img_name = gt_name.replace('_mask','')
            image_path = os.path.join(args.img_path, img_name)
            dsm_path = os.path.join(args.dsm_path, img_name)
            if not os.path.exists(image_path):
                continue
            gt_, box, point, point_label_list,img_embedding, dsm_embedding,resized_size_be_padding, image_ori_size, new_lab_list = process(gt_name, img_name, mode)
            if gt_ is not None:
                np.savez_compressed(
                    join(save_path, gt_name.split('.')[0] + ".npz"),
                    label_except_bk=box,
                    point=point,
                    point_label_list = point_label_list,
                    gts=gt_,
                    img_embeddings=img_embedding,
                    dsm_embedding=dsm_embedding,
                    image_shape=image_ori_size,
                    resized_size_before_padding=resized_size_be_padding,
                    new_lab_list = new_lab_list
                )
        print("Num. of processed train images (delete images with no any targets):", len(imgs))
