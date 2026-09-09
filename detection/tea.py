'''
在
/workspace/songzhengyao/bdzoo_cifar10/detection/bit_flip_fixed_flip_k_layer_feature_difference_unconsider_dataset_4-8bits_anlyse_optimize_mask_only_element0_flip_energy_scores_adoptive_opti_epochs_improve_local_optimal.py
基础上做了一下改进：
1. 在优化mask的时候，也考虑只flip 0元素这个限制。
2. final_mask = (torch.sigmoid(mask)).float() 取消设定大于0.5的限制
'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append(os.getcwd())
from pprint import  pformat
import yaml
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from defense.base import defense
import scipy
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.nCHW_nHWC import *

import tqdm
import heapq
from PIL import Image
from utils.bd_dataset_v2 import dataset_wrapper_with_transform,xy_iter, prepro_cls_DatasetBD_v2
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from collections import Counter
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
import random
import csv
from sklearn import metrics
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset 
import torchvision
from collections import OrderedDict
import matplotlib.pyplot as plt




 

class Bit(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv
        
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        
        self.args = args
 

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
         
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/bit/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--clean_sample_num', type=int, default=20, help="reference_dataset")
        parser.add_argument('--target_layer', type=str, required=True)
        # 新增翻转强度参数
        parser.add_argument('--opti_epoch', type=int, default=5, help='mask optimization epochs')
        parser.add_argument("--interval", type=int,default=20,help="Interval of epochs to load checkpoints (e.g., 20 -> 19,39,...)")

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/bit_pretrain/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'detection_info/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
                
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')
 



 

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model = model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        self.device = self.args.device

    def cal(self, true, pred):
        TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
        return TN, FP, FN, TP 
    
    def metrix(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN) if (TP+FN) != 0 else 0.0
        FPR = FP/(FP+TN) if (FP+TN) != 0 else 0.0
        precision = TP/(TP+FP) if (TP+FP) != 0 else 0.0
        acc = (TP+TN)/(TN+FP+FN+TP) if (TN+FP+FN+TP) != 0 else 0.0
        return TPR, FPR, precision, acc
    
    @staticmethod
    def tensor_to_uint8(x, mean, std):
        """
        x: torch.Tensor, [B,C,H,W] or [C,H,W], normalized
        return: uint8 tensor in [0,255]
        """
        device = x.device
        mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        std  = torch.tensor(std,  device=device).view(1, -1, 1, 1)

        if x.dim() == 3:
            mean, std = mean[0], std[0]

        x = x * std + mean        # de-normalize → [0,1]
        x = (x * 255.0).round()   # → [0,255]
        x = x.clamp(0, 255).to(torch.uint8)
        return x 
    
   

    @staticmethod
    def reconstruct_image(bit_planes):

        device = bit_planes.device

        weights = (2 ** torch.arange(8, dtype=torch.float32, device=device)).reshape(1, 1, -1, 1, 1)

        img_recon = torch.sum(bit_planes * weights, dim=2)
        return img_recon

    @staticmethod
    def batch_uint8_to_tensor(batch_tensor_input_255, transform):

        B, C, H, W = batch_tensor_input_255.shape
        output_list = []
        
        for i in range(B):
            img = batch_tensor_input_255[i]  # [C,H,W]

            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.squeeze(img_np).astype(np.uint8)
            
            # 转为PIL Image并应用变换
            img_pil = Image.fromarray(img_np)
            # img.save("/workspace/songzhengyao/bdzoo_cifar10/bit_plane/output.png") 存下翻转后的图片
            img_transformed = transform(img_pil)
            output_list.append(img_transformed)
        
        return torch.stack(output_list, dim=0)

    
    # @staticmethod
    # def batch_uint8_to_tensor_for_gradient(batch_tensor_input_255, transform):
    #     """
    #     将0-255的张量转为模型输入的归一化张量
    #     batch_tensor_input_255: [B,C,H,W] uint8
    #     transform: 数据变换（ToTensor + Normalize）
    #     return: [B,C,H,W] float32 归一化张量
    #     """
    #     B, C, H, W = batch_tensor_input_255.shape
    #     output_list = []

    #     for i in range(B):
    #         img = batch_tensor_input_255[i]  # [C,H,W]

    #         # 转为 [H,W,C] 并去归一化
    #         # img_np = img.permute(1, 2, 0).cpu().numpy()
    #         img_np = img.permute(1, 2, 0).detach().cpu().numpy()
    #         img_np = np.squeeze(img_np).astype(np.uint8)
    #                                                             ###
    #         # 转为PIL Image并应用变换
    #         img_pil = Image.fromarray(img_np)
    #         img_transformed = transform(img_pil)
    #         output_list.append(img_transformed)
        
    #     return torch.stack(output_list, dim=0)

    @staticmethod
    def batch_uint8_to_tensor_for_gradient(batch_tensor_input_255, transform):
        """
        输入： batch_tensor_input_255: [B,C,H,W] uint8
        输出：归一化后的张量 [B,C,H,W]
        """
        B, C, H, W = batch_tensor_input_255.shape
        device = batch_tensor_input_255.device
        # 查找 Normalize(mean, std)
        mean = None
        std = None
        for t in transform.transforms:
            if isinstance(t, torchvision.transforms.Normalize):
                mean = torch.tensor(t.mean).view(1, C, 1, 1).to(device)
                std = torch.tensor(t.std).view(1, C, 1, 1).to(device)

        # 归一化
        img = batch_tensor_input_255.float() / 255.0
        img_input = (img - mean) / std  # 全向量化广播
        
        return img_input



    @staticmethod
    def load_models_by_interval(args):
        model_list = []
        epoch_list = []

        base_path = os.path.join("record", args.result_file)

        interval = args.interval
        max_epoch = args.epochs

        epochs = list(range(interval - 1, max_epoch, interval))

        for epoch in epochs:
            pt_path = os.path.join(base_path, f"attack_epoch_{epoch}.pt")

            if not os.path.exists(pt_path):
                print(f"⚠️ skip missing: {pt_path}")
                continue

            checkpoint = torch.load(pt_path, map_location="cpu")

            model = generate_cls_model(args.model, args.num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])

            model_list.append(model)
            epoch_list.append(epoch)

        return model_list, epoch_list



    def filtering(self):
        start = time.perf_counter()
        self.set_devices()
        fix_random(self.args.random_seed)

        ### 1. 加载模型
        models, model_epochs = Bit.load_models_by_interval(self.args)
        device = self.args.device
        if "," in device:
            device = f"cuda:{device[5:].split(',')[0]}"
        for i in range(len(models)):
            models[i] = models[i].to(device)
            models[i].eval()

        # # 多卡/单卡设置
        # if "," in self.device:
        #     for i in range(len(models)):
        #         models[i] = torch.nn.DataParallel(
        #             models[i],
        #             device_ids=[int(i) for i in self.args.device[5:].split(",")]
        #         )
        #         self.args.device = f'cuda:{models[i].device_ids[0]}'
        #         models[i].to(self.args.device)
        #         models[i].eval()


        




        
        ### 2. 加载数据和变换
        test_tran = get_transform(self.args.dataset, self.args.input_height, self.args.input_width, train = False)
        
        # 后门训练集
        bd_train_dataset = self.result['bd_train'].wrapped_dataset
        pindex = np.where(np.array(bd_train_dataset.poison_indicator) == 1)[0]  # 真实后门样本索引

        # result_model = generate_cls_model(self.args.model, self.args.num_classes)
        # result_model.load_state_dict(self.result['model'])
        # module_dict = dict(result_model.named_modules())
        # if self.args.target_layer not in module_dict:
        #     logging.error(f"Target layer '{self.args.target_layer}' not found among model.named_modules(). Available keys: {list(module_dict.keys())}")
        #     raise ValueError(f"target_layer '{self.args.target_layer}' not found")
        # target_layer = module_dict[self.args.target_layer]



        # 干净测试集（用于计算阈值）
        clean_test_dataset = self.result['clean_test'].wrapped_dataset
        images = []
        labels = []
        for img, label in clean_test_dataset:
            images.append(img)
            labels.append(label)
        
        # 采样干净样本（按类别均衡采样）
        class_idx_whole = []
        num_per_class = max(1, int(self.args.clean_sample_num / self.args.num_classes))
        for i in range(self.args.num_classes):
            class_indices = np.where(np.array(labels)==i)[0]
            if len(class_indices) > 0:
                class_idx_whole.append(np.random.choice(class_indices, num_per_class, replace=False))
        
        class_idx_whole = np.concatenate(class_idx_whole, axis=0) if class_idx_whole else np.array([])
        image_c = [images[i] for i in class_idx_whole] if len(class_idx_whole) > 0 else []
        label_c = [labels[i] for i in class_idx_whole] if len(class_idx_whole) > 0 else []
        clean_test_dataset = xy_iter(image_c, label_c, transform=test_tran)
        clean_set_loader = DataLoader(clean_test_dataset, batch_size=64, shuffle=False, pin_memory=self.args.pin_memory)

            # 待检测的训练集
        images_poison = []
        labels_poison = []
        for img, label,*other_info in bd_train_dataset:
            images_poison.append(img)
            labels_poison.append(label)        
        train_dataset = xy_iter(images_poison, labels_poison, transform=test_tran)
        train_dataset_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=self.args.pin_memory)
            
      
    
        clean_features_before = []
        clean_features_after = [] 
        clean_feature_similarity = []   
        clean_feature_distance = []
        clean_scores = []
        clean_features_proj = []
        clean_features_Chebyshev = []
        

        with torch.no_grad():
            for _input, _label in clean_set_loader:
                _input, _label = _input.to(self.args.device), _label.to(self.args.device)
                                

                ##########################################
                ##试试energy score
                ##########################################
                energy_list = []
                for model in models:
                    logits_after = model(_input)
                    energy = -torch.logsumexp(logits_after, dim=1)
                    energy_list.append(energy)
                energy_stack = torch.stack(energy_list, dim=0) 
                score = energy_stack.sum(dim=0)

                 
                clean_scores.extend(score.tolist())

        # clean_features_Chebyshev_sorted = torch.sort(torch.cat(clean_features_Chebyshev, dim=0), descending=False)[0]
        # clean_features_proj_sorted = torch.sort(torch.cat(clean_features_proj, dim=0), descending=False)[0] ##descending=False 从小到大
        
        # clean_scores_sorted = torch.sort(torch.cat(clean_scores, dim=0), descending=False)[0]
        clean_scores_sorted = torch.sort(torch.tensor(clean_scores), descending=False)[0]

        # clean_feature_distance_sorted = torch.sort(torch.cat(clean_feature_distance, dim=0), descending=False)[0]
        # clean_feature_similarity_sorted = torch.sort(torch.cat(clean_feature_similarity, dim=0), descending=False)[0]

        # threshold_high = float(clean_feature_similarity_sorted[int(0.95 * len(clean_feature_similarity_sorted))])  # 95%分位数（大值），正常干净的是不相似的，值小
        # threshold_low = float('-inf')
        # threshold_low = float('inf')
        # threshold_high = float(clean_features_Chebyshev_sorted[int(0 * len(clean_features_Chebyshev_sorted))]) # 0%分位数（小值），后门值小 干净的大
        threshold_low = float('inf')
        threshold_high = float(clean_scores_sorted[int(0.95 * len(clean_scores_sorted)-1)]) # 95%分位数（大值），后门值大 干净的小
        logging.info(f"Threshold low: {threshold_low}, threshold high: {threshold_high}")


        # #########################################
        # ##分析backdoored samples，挑出来backdoored samples
        # #########################################



        ### 5. 检测待检测样本
        inspection_features_before = []
        inspection_features_after = [] 
        inspection_feature_similarity = []   #  
        inspection_features_Chebyshev = []
        inspection_scores = []

        # ============================================================
        # Logits scale analysis across checkpoints / epochs
        # Goal: check whether the overall logits scale is comparable
        # across all loaded models on the SAME inspection samples.
        # Statistics are accumulated online to avoid storing all logits.
        # ============================================================
        logits_scale_stats = [
            {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "sum_abs": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
            for _ in models
        ]

        with torch.no_grad():
            for _input, _label in train_dataset_loader:
                _input, _label = _input.to(self.args.device), _label.to(self.args.device)

                ##########################################
                ## Energy score + logits scale statistics
                ##########################################
                energy_list = []
                for model_idx, model in enumerate(models):
                    logits_after = model(_input)

                    # ---- accumulate logits scale statistics ----
                    logits_flat = logits_after.detach().float().reshape(-1)
                    stat = logits_scale_stats[model_idx]
                    stat["count"] += logits_flat.numel()
                    stat["sum"] += logits_flat.sum().item()
                    stat["sum_sq"] += torch.sum(logits_flat * logits_flat).item()
                    stat["sum_abs"] += logits_flat.abs().sum().item()
                    stat["min"] = min(stat["min"], logits_flat.min().item())
                    stat["max"] = max(stat["max"], logits_flat.max().item())

                    energy = -torch.logsumexp(logits_after, dim=1)
                    energy_list.append(energy)
                energy_stack = torch.stack(energy_list, dim=0) 
                score = energy_stack.sum(dim=0)



                # 翻转后的预测
                # logits_after = model(img_input)
                # logits_after_2 = model_2(img_input)
                # logits_after_3 = model_3(img_input)

                # score = -torch.logsumexp(logits_after, dim=1) - torch.logsumexp(logits_after_2, dim=1) - torch.logsumexp(logits_after_3, dim=1)
                
                # inspection_scores.append(score.cpu())
                inspection_scores.extend(score.tolist())

        # ============================================================
        # Save logits scale statistics for each checkpoint / epoch
        # ============================================================
        logits_scale_rows = []
        for model_idx, stat in enumerate(logits_scale_stats):
            n = max(stat["count"], 1)
            mean_val = stat["sum"] / n
            mean_sq = stat["sum_sq"] / n
            var_val = max(mean_sq - mean_val * mean_val, 0.0)
            std_val = var_val ** 0.5
            abs_mean_val = stat["sum_abs"] / n
            rms_val = mean_sq ** 0.5
            range_val = stat["max"] - stat["min"]

            epoch = model_epochs[model_idx] if model_idx < len(model_epochs) else model_idx
            logits_scale_rows.append([
                epoch,
                stat["min"],
                stat["max"],
                range_val,
                mean_val,
                std_val,
                abs_mean_val,
                rms_val,
            ])

        logits_scale_csv = os.path.join(self.args.save_path, 'logits_scale_statistics.csv')
        with open(logits_scale_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'logits_min', 'logits_max', 'logits_range',
                'logits_mean', 'logits_std', 'logits_abs_mean', 'logits_rms'
            ])
            writer.writerows(logits_scale_rows)

        # Quantify cross-model scale consistency. RMS and std are more
        # informative than raw min/max because they are less sensitive
        # to a few extreme logits. A ratio close to 1 means similar scale.
        rms_values = np.asarray([row[7] for row in logits_scale_rows], dtype=np.float64)
        std_values = np.asarray([row[5] for row in logits_scale_rows], dtype=np.float64)
        abs_mean_values = np.asarray([row[6] for row in logits_scale_rows], dtype=np.float64)

        def _safe_scale_ratio(values):
            values = values[np.isfinite(values)]
            values = values[values > 0]
            if len(values) == 0:
                return float('nan')
            return float(values.max() / values.min())

        rms_ratio = _safe_scale_ratio(rms_values)
        std_ratio = _safe_scale_ratio(std_values)
        abs_mean_ratio = _safe_scale_ratio(abs_mean_values)

        logits_scale_summary = os.path.join(self.args.save_path, 'logits_scale_summary.txt')
        with open(logits_scale_summary, 'w', encoding='utf-8') as f:
            f.write('Overall logits-scale consistency across loaded models\n')
            f.write('=====================================================\n')
            f.write(f'Number of models: {len(models)}\n')
            f.write(f'Epochs: {model_epochs}\n')
            f.write(f'RMS max/min ratio: {rms_ratio:.6f}\n')
            f.write(f'Std max/min ratio: {std_ratio:.6f}\n')
            f.write(f'Abs-mean max/min ratio: {abs_mean_ratio:.6f}\n')
            f.write('Interpretation: ratios closer to 1 indicate more similar logits scale.\n')

        logging.info(
            'Logits scale | RMS max/min: %.4f | Std max/min: %.4f | AbsMean max/min: %.4f',
            rms_ratio, std_ratio, abs_mean_ratio
        )

        # Plot the global logits scale over epochs.
        # Mean +/- std reflects the center and spread of all logits;
        # min/max are included as light boundary curves.
        epochs_plot = np.asarray([row[0] for row in logits_scale_rows])
        mins_plot = np.asarray([row[1] for row in logits_scale_rows])
        maxs_plot = np.asarray([row[2] for row in logits_scale_rows])
        means_plot = np.asarray([row[4] for row in logits_scale_rows])
        stds_plot = np.asarray([row[5] for row in logits_scale_rows])

        plt.figure(figsize=(7, 5))
        plt.plot(epochs_plot, means_plot, marker='o', label='Mean logits')
        plt.fill_between(
            epochs_plot,
            means_plot - stds_plot,
            means_plot + stds_plot,
            alpha=0.2,
            label='Mean ± Std'
        )
        plt.plot(epochs_plot, mins_plot, linestyle='--', linewidth=1, label='Min logits')
        plt.plot(epochs_plot, maxs_plot, linestyle='--', linewidth=1, label='Max logits')
        plt.xlabel('Epoch')
        plt.ylabel('Logit value')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.args.save_path, 'logits_scale.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


        # inspection_features_Chebyshev = torch.cat(inspection_features_Chebyshev,dim=0)
        # inspection_feature_similarity = torch.cat(inspection_feature_similarity, dim=0)

        # inspection_scores = torch.cat(inspection_scores, dim=0)
        inspection_scores = torch.tensor(inspection_scores)
 
        # suspicious_mask = inspection_feature_similarity > threshold_high
        # suspicious_mask = inspection_features_Chebyshev <= threshold_high
        suspicious_mask = inspection_scores >= threshold_high
        suspicious_indices = torch.where(suspicious_mask)[0].tolist()
        # train_scores = inspection_feature_similarity.numpy()  # 用于AUROC计算
        # train_scores = -inspection_features_Chebyshev.numpy()
        train_scores = inspection_scores
        
        logging.info(f"Flagged {len(suspicious_indices)} suspicious samples out of {len(train_scores)}")
        end = time.perf_counter()
        time_minute = (end - start) / 60.0
        ### 6. 评估检测效果
        if len(suspicious_indices) == 0 and len(pindex) == 0:
            # 干净模型
            with open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['record', 'TN', 'FP', 'FN', 'TP', 'TPR', 'FPR', 'target'])
                csv_write.writerow([self.args.result_file, 'This is clean model'])
        else:
            # 构建真实标签和预测标签
            true_labels = np.zeros(len(images_poison), dtype=int)
            true_labels[pindex] = 1  # 后门样本标记为1
            
            pred_labels = np.zeros(len(images_poison), dtype=int)
            pred_labels[suspicious_indices] = 1  # 检测出的可疑样本标记为1
            
            # 计算指标
            tn, fp, fn, tp = self.cal(true_labels, pred_labels)
            TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)
            
            # 计算AUROC
            # fpr_auroc, tpr_auroc, _ = roc_curve(true_labels, train_scores)
            fpr_auroc, tpr_auroc, thresholds = roc_curve(true_labels, train_scores)
            roc_auc = auc(fpr_auroc, tpr_auroc)

            
             
    def detection(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        self.filtering()
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bit Plane Flip Backdoor Detection (Idea 2)")
    Bit.add_arguments(parser)
    args = parser.parse_args()
    
    # 设置默认值
    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'
    

    
    # 初始化并运行检测
    bit_method = Bit(args)
    result = bit_method.detection(args.result_file)
