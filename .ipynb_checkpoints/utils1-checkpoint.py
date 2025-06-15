import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
from torchvision.transforms.functional import InterpolationMode
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from utils.get_prompts import find_all_info
import matplotlib.pyplot as plt
from PIL import Image
# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")
import glob
# add arguments to the parser
parser.add_argument(
    "-i",
    "--img_path",
    type=str,
    default=f"data/test_data",
    help="path to the images",
)
parser.add_argument(
    "-gt",
    "--gt_path",
    type=str,
    default=f"data/test_data",
    help="path to the ground truth (gt)",
)

parser.add_argument(
    "-task",
    "--task_name",
    type=str,
    default=f"22_Heart",
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
    default=f"data_vaihin_cropped_200",
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
    default="sam_vit_b_01ec64.pth",
    help="original sam checkpoint",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()

# Parameters
## SwinFusion
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 
# BATCH_SIZE = 4 # For backbone ViT-Huge

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

MODEL = 'UNetformer'
# MODEL = 'FTUNetformer'
MODE = 'Train'
# MODE = 'Test'
DATASET = 'Potsdam'
# DATASET = 'Vaihingen'
IF_SAM = True
# IF_SAM = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    Stride_Size = 32
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
elif DATASET == 'Potsdam':
    train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
                '4_12', '6_8', '6_12', '6_7', '4_11']
    test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
    Stride_Size = 128
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

print(MODEL + ', ' + MODE + ', ' + DATASET + ', IF_SAM: ' + str(IF_SAM) + ', WINDOW_SIZE: ', WINDOW_SIZE, 
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size),
      ', epochs: ' + str(epochs), ', save_epoch: ', str(save_epoch),)
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, device = args.device)
# ResizeLongestSide (1024), including image and gt
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, mode, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.ids = ids
        self.mode = mode
        # List of files
        if mode == "train":
            self.data_files = [DATA_FOLDER.format(id) for id in ids]
            self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
            self.label_files = [LABEL_FOLDER.format(id) for id in ids]
        else:
            images_folder = os.path.join(FOLDER, "valid_data/images")
            dsm_folder = os.path.join(FOLDER, "valid_data/dsm")
            label_folder = os.path.join(FOLDER, "valid_data/labels")
            # 支持常见的图片格式
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
            # 读取images文件夹中的所有图片
            self.data_files = []
            for ext in image_extensions:
                self.data_files.extend(glob.glob(os.path.join(images_folder, ext)))
                self.data_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
                
            # 读取dsm文件夹中的所有图片
            self.dsm_files = []
            for ext in image_extensions:
                self.dsm_files.extend(glob.glob(os.path.join(dsm_folder, ext)))
                self.dsm_files.extend(glob.glob(os.path.join(dsm_folder, ext.upper())))
            # 读取label文件夹中的所有图片
            self.label_files = []
            for ext in image_extensions:
                self.label_files.extend(glob.glob(os.path.join(label_folder, ext)))
                self.label_files.extend(glob.glob(os.path.join(label_folder, ext.upper())))
                
        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}
        self.resize_transform = ResizeLongestSide(target_length=1024)

    def __len__(self):
        if self.mode == "train":
            if DATASET == 'Potsdam':
                if self.ids == train_ids:
                    return 1 * 5000
                if self.ids == test_ids:
                    return 1 * 1600
            elif DATASET == 'Vaihingen':
                if self.ids == train_ids:
                    return 1 * 1
                if self.ids == test_ids:
                    return 1 * 1600
            else:
                return None
        else:
            return len(self.data_files)
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        if self.mode == "train":
            random_idx = random.randint(0, len(self.data_files) - 1)
            # If the tile hasn't been loaded yet, put in cache
            if random_idx in self.data_cache_.keys():
                data = self.data_cache_[random_idx]
            else:
                # Data is normalized in [0, 1]
                ## Potsdam IRRG
                if DATASET == 'Potsdam':
                    ## RGB
                    data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                    ## IRRG
                    # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                    data = 1 / 255 * np.asarray(data, dtype='float32')
                else:
                ## Vaihingen IRRG
                    data = io.imread(self.data_files[random_idx])
                    data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
                if self.cache:
                    self.data_cache_[random_idx] = data

            if random_idx in self.dsm_cache_.keys():
                dsm = self.dsm_cache_[random_idx]
            else:
                # DSM is normalized in [0, 1]
                dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
                min = np.min(dsm)
                max = np.max(dsm)
                smooth = 0
                if max == 0:
                    smooth = 1
                dsm = (dsm - min) / (max - min + smooth)
                if self.cache:
                    self.dsm_cache_[random_idx] = dsm

            if random_idx in self.label_cache_.keys():
                label = self.label_cache_[random_idx]
            else:
                # Labels are converted from RGB to their numeric values
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
                if self.cache:
                    self.label_cache_[random_idx] = label

            # Get a random patch
            x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
            data_p = data[:, x1:x2, y1:y2]
            dsm_p = dsm[x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]
        else:
            # print(self.data_files[i])
            # print(self.dsm_files[i])
            # print(self.label_files[i])
            data = io.imread(self.data_files[i]).transpose((2, 0, 1))
            data_p = (1 / 255.0) * np.asarray(data, dtype='float32')
            dsm = np.asarray(io.imread(self.dsm_files[i]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            smooth = 0
            if max == 0:
                smooth = 1
            dsm_p = (dsm - min) / (max - min + smooth)
            label_p = np.asarray(io.imread(self.label_files[i]), dtype='int64')
        # print(data_p.shape)
        # 在调用 SAM 转换前使用此函数
        
        # data_p = prepare_image_for_pil(data_p.transpose((1, 2, 0)))
        # dsm_p = prepare_image_for_pil(dsm_p.transpose((1, 0)))
        # data_p = sam_transform.apply_image(data_p, interpolation=InterpolationMode.BILINEAR)
        # dsm_p = sam_transform.apply_image(dsm_p, interpolation=InterpolationMode.BILINEAR)
        # data_p = sam_transform.apply_image(data_p, interpolation=InterpolationMode.BILINEAR)
        # Data augmentation
        if self.ids == train_ids:
            data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)
        # data_p = F.interpolate(data_p, size=(1024, 1024), mode='nearest', align_corners=None)
        # dsm_p = F.interpolate(data_p, size=(1024, 1024), mode='nearest', align_corners=None)
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
        data_p = upsample_patch(torch.tensor(data_p))
        dsm_p = upsample_patch(torch.tensor(dsm_p))
        # print(dsm_p.shape)
        gt = label_p.copy()
        del_lab = []
        if np.unique(label_p)[0] != 0:
            label_list = np.unique(label_p)
            for label in label_list:
                gt_single = (label_p == label) + 0
                if np.sum(gt_single) <1:
                    gt[gt == label] = 0
                    del_lab.append(label)
                    assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0]) , \
                    f"标签数量不匹配！处理后标签: {np.unique(gt)}, 删除标签: {del_lab}, 原始标签: {np.unique(label_p)}"
        else:
            label_list = np.unique(label_p)[0:]
            for label in label_list:
                gt_single = (label_p == label) + 0
                if np.sum(gt_single) <1:
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
            point_list = np.zeros((1, 150, 2), dtype=np.uint8)
            point_label_list = np.zeros((1, 150), dtype=np.uint8)
        else:
            # box_list = np.zeros((gt_.shape[0], 4), dtype=np.uint8)
            point_list, point_label_list, box_list = find_all_info(mask=gt_, 
                                           label_list=new_lab_list, 
                                           point_num=150)
            # print(point_list.shape)
            # print(point_label_list)
        # convert img embedding, gt_mask, bounding box to torch tensor
        box = torch.tensor(box_list).float() # B, 4
        box = self.resize_transform.apply_boxes_torch(box, (gt_.shape[-2], gt_.shape[-1]))
        
        point = torch.tensor(point_list).float() # B, 4
        point_label_list = torch.tensor(point_label_list).float()
        point = self.resize_transform.apply_coords_torch(point, (gt_.shape[-2], gt_.shape[-1]))
        # visualize_and_save(data_p, random_idx, x1, y1)
        return {"img_embed": data_p,
                "dsm_embed":dsm_p,
                "gt2D": torch.tensor(gt_),
                "box": box,
                "point": point,
                "point_label": point_label_list,
                "image_ori_size": [256, 256],
                "size_before_pad": [1024, 1024]
        }
        # Return the torch.Tensor values
        # return (torch.from_numpy(data_p),
        #         torch.from_numpy(dsm_p),
        #         torch.from_numpy(label_p))

## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def visualize_and_save(data_p, random_idx, x1, y1, save_dir="tmp_img", name_prefix="sample"):
    """可视化并保存图像补丁"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将 PyTorch 张量转换为 NumPy 数组
    if isinstance(data_p, torch.Tensor):
        # 如果张量在 GPU 上，先移到 CPU
        if data_p.is_cuda:
            data_p = data_p.cpu()
        # 移除批次维度（如果有）并转换为 NumPy
        img = data_p.detach().numpy()  # 形状: [C, H, W]
    else:
        img = data_p  # 假设已经是 NumPy 数组
    
    # 调整通道顺序：[C,H,W] → [H,W,C]
    img = np.transpose(img, (1, 2, 0))
    
    # 如果数据已归一化到 [0,1]，转回 [0,255]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # 保存图像
    save_path = os.path.join(save_dir, f"{name_prefix}_{random_idx}_{x1}_{y1}.png")
    Image.fromarray(img).save(save_path)
    
    # 可选：使用 matplotlib 显示图像
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)
#     plt.title(f"Patch from Image {random_idx} at ({x1},{y1})")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"{name_prefix}_{random_idx}_{x1}_{y1}_vis.png"))
#     plt.close()
    
    print(f"图像已保存至: {save_path}")
    
def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

# def visualize_and_save(data_p, random_idx, x1, y1, save_dir="tmp_img", name_prefix="sample"):
#     """可视化并保存图像补丁"""
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 确保保存目录存在
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # 转换数据格式：[C,H,W] → [H,W,C]
#     img = data_p.transpose(1, 2, 0)  # 调整通道顺序
    
#     # 如果数据已归一化到 [0,1]，转回 [0,255]
#     if img.max() <= 1.0:
#         img = (img * 255).astype(np.uint8)
    
#     # 保存图像
#     save_path = os.path.join(save_dir, f"{name_prefix}_{random_idx}_{x1}_{y1}.png")
#     Image.fromarray(img).save(save_path)
    
#     # 可选：使用matplotlib显示图像
# #     plt.figure(figsize=(10, 10))
# #     plt.imshow(img)
# #     plt.title(f"Patch from Image {random_idx}")
# #     plt.axis('off')
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(save_dir, f"{name_prefix}_{random_idx}_{x1}_{y1}_vis.png"))
# #     plt.close()
    
#     print(f"图像已保存至: {save_path}")

def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU
