"""
三模态训练脚本 - 用于 MOSEI/MOSI 数据集
实现三模态 beta 计算逻辑:
- 最强模态 beta: gap = (score_最强 - score_次强 + score_最强 - score_最弱) / 2
- 次强模态 beta: gap = (score_次强 - score_最弱)
- 最弱模态 beta = 0

模型架构:
- Text: Transformer 编码器 (来自 mosei 项目)
- Audio: MLP 编码器
- Visual: MLP 编码器

训练设置 (参考 mosei 项目):
- batch_size: 32
- lr: 0.001
- n_epochs: 30
- optimizer: SGD (momentum=0.9, weight_decay=1e-4)
- scheduler: cosine_schedule_with_warmup (warmup=2.5 epochs)
"""

import argparse
import os
import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
import pickle
import csv

# Cosine scheduler with warmup (from transformers library)
try:
    from transformers import get_cosine_schedule_with_warmup
except ImportError:
    # 如果没有安装 transformers，使用简单的 cosine scheduler
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        from torch.optim.lr_scheduler import LambdaLR
        import math
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return LambdaLR(optimizer, lr_lambda, last_epoch)

# 支持两种模型: 简单的 TriModalClassifier 或 Transformer-based 的 TriModalTransformer
from models.basic_model import TriModalClassifier
from models.transformers import TriModalTransformer
from MOSEIDataset import MOSEIDataset, MOSEIDatasetCached, get_mosei_dataloaders
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MOSEI', type=str,
                        choices=['MOSEI', 'MOSI'])
    parser.add_argument('--modulation', default='Ours', type=str,
                        choices=['Normal', 'OGM', 'Ours'])
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--dataset_name', default='mosei_senti', type=str)
    
    # ========== 训练超参数 (参考 mosei 项目) ==========
    parser.add_argument('--batch_size', default=32, type=int)  # mosei: 32
    parser.add_argument('--epochs', default=30, type=int)       # mosei: 30
    parser.add_argument('--n_classes', default=3, type=int)
    
    # 模态特征维度 (MOSEI: text=300, audio=74, visual=35)
    parser.add_argument('--text_dim', default=300, type=int)
    parser.add_argument('--audio_dim', default=74, type=int)
    parser.add_argument('--visual_dim', default=35, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int, help='MLP hidden dim')
    parser.add_argument('--embed_dim', default=120, type=int, help='embedding dimension (mosei: 120)')
    
    # Transformer 配置 (参考 mosei: depth=4, num_heads=10, embed_dim=120)
    parser.add_argument('--transformer_depth', default=4, type=int, help='Transformer layers')
    parser.add_argument('--transformer_heads', default=10, type=int, help='attention heads (mosei: 10)')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in Transformer')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--fusion_method', default='concat', type=str, choices=['concat', 'gated'])
    
    # 模型选择
    parser.add_argument('--model_type', default='transformer', type=str,
                        choices=['simple', 'transformer'],
                        help='simple: TriModalClassifier, transformer: TriModalTransformer')

    # ========== 优化器设置 (参考 mosei: SGD, lr=0.001, momentum=0.9, weight_decay=1e-4) ==========
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])  # mosei: SGD
    parser.add_argument('--learning_rate', default=0.001, type=float)  # mosei: 0.001
    parser.add_argument('--momentum', default=0.9, type=float)         # mosei: 0.9
    parser.add_argument('--weight_decay', default=1e-4, type=float)    # mosei: 1e-4
    parser.add_argument('--warmup_epochs', default=2.5, type=float, help='warmup epochs for cosine scheduler')

    parser.add_argument('--modulation_starts', default=0, type=int)
    parser.add_argument('--modulation_ends', default=30, type=int)  # 与 epochs 对齐
    parser.add_argument('--alpha', default=0.8, type=float, help='proximal 正则化系数')
    parser.add_argument('--beta_scale', default=0.8, type=float, help='beta 缩放系数')
    parser.add_argument('--k_threshold', default=0.1, type=float, help='prime learning window 的 k 值阈值')
    
    parser.add_argument('--ckpt_path', default='./ckpt', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--use_tensorboard', default=True, type=bool)
    parser.add_argument('--tensorboard_path', default='./logs', type=str)
    parser.add_argument('--random_seed', default=2024, type=int)  # mosei: 2024
    parser.add_argument('--gpu_ids', default='0', type=str)

    return parser.parse_args()


def compute_trimodal_beta(score_t, score_a, score_v, alpha=0.95):
    """
    计算三模态的 beta 值
    
    逻辑:
    - 先判断出最强、次强和最弱的模态
    - 最强模态: gap = (score_最强 - score_次强 + score_最强 - score_最弱) / 2
    - 次强模态: gap = (score_次强 - score_最弱)
    - 最弱模态: beta = 0
    
    Args:
        score_t: 文本模态的 score (sum of softmax probabilities at true labels)
        score_a: 音频模态的 score
        score_v: 视觉模态的 score
        alpha: 缩放系数
    
    Returns:
        beta_t, beta_a, beta_v: 三个模态的 beta 值
    """
    tanh = torch.tanh
    
    # 将 scores 放入列表进行排序
    scores = [
        ('text', score_t),
        ('audio', score_a),
        ('visual', score_v)
    ]
    
    # 按 score 降序排序
    scores_sorted = sorted(scores, key=lambda x: x[1].item(), reverse=True)
    
    strongest_name, strongest_score = scores_sorted[0]
    middle_name, middle_score = scores_sorted[1]
    weakest_name, weakest_score = scores_sorted[2]
    
    # 计算各模态的 gap 和 beta
    # 最强模态: gap = (score_最强 - score_次强 + score_最强 - score_最弱) / 2
    gap_strongest = (strongest_score - middle_score + strongest_score - weakest_score) / 2
    beta_strongest = alpha * torch.exp(tanh(gap_strongest))
    
    # 次强模态: gap = (score_次强 - score_最弱)
    gap_middle = middle_score - weakest_score
    beta_middle = alpha * torch.exp(tanh(gap_middle))
    
    # 最弱模态: beta = 0
    beta_weakest = torch.tensor(0.0).to(score_t.device)
    
    # 按原始顺序返回 beta 值
    beta_dict = {
        strongest_name: beta_strongest,
        middle_name: beta_middle,
        weakest_name: beta_weakest
    }
    
    return beta_dict['text'], beta_dict['audio'], beta_dict['visual']


def train_epoch_trimodal(args, epoch, model, device, dataloader, optimizer, scheduler, 
                         writer=None, global_model=None, 
                         text_trace_list=None, audio_trace_list=None, visual_trace_list=None):
    """
    三模态训练一个 epoch
    
    新增参数:
        text_trace_list: 文本模态的 FIM trace 列表 (用于计算 k)
        audio_trace_list: 音频模态的 FIM trace 列表
        visual_trace_list: 视觉模态的 FIM trace 列表
    """
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    tanh = nn.Tanh()
    
    # ==== 计算 k 值 (使用最强模态的 FIM trace 来判断 prime learning window) ====
    # 使用 10 个 epoch 的平滑来判断是否处于 prime learning window
    # 选择最强模态（FIM trace 最大的模态）
    if epoch == 0 or text_trace_list is None or len(text_trace_list) < 11:
        k = 1.0  # 初始阶段，认为处于 prime learning window
        strongest_modality = 'text'
    else:
        # 获取各模态最近的 FIM trace 平均值
        tr_text = sum(text_trace_list[-10:]) / 10
        tr_audio = sum(audio_trace_list[-10:]) / 10
        tr_visual = sum(visual_trace_list[-10:]) / 10
        
        # 找到最强模态 (FIM trace 最大的)
        traces = {'text': tr_text, 'audio': tr_audio, 'visual': tr_visual}
        strongest_modality = max(traces, key=traces.get)
        
        # 使用最强模态的 trace 来计算 k
        if strongest_modality == 'text':
            trace_list = text_trace_list
        elif strongest_modality == 'audio':
            trace_list = audio_trace_list
        else:
            trace_list = visual_trace_list
        
        tr1 = sum(trace_list[-10:]) / 10
        tr2 = sum(trace_list[-11:-1]) / 10
        
        # 计算 k: FIM 变化率
        if tr1 != 0:
            k = (tr1 - tr2) / tr1
        else:
            k = 1.0
    
    print("----------------------------")
    print(f"k 的值为 {k:.6f} (基于最强模态 '{strongest_modality}' 的 FIM trace)")
    in_prime_window = k > args.k_threshold if hasattr(args, 'k_threshold') else k > 0.05
    print(f"Prime Learning Window: {'是' if in_prime_window else '否'} (阈值: {getattr(args, 'k_threshold', 0.05)})")
    print("----------------------------")
    
    if global_model is None:
        global_model = cp.deepcopy(model)
    
    model.train()
    print(f"Start training epoch {epoch}...")
    
    # 记录参数名
    record_names_text = []
    record_names_audio = []
    record_names_visual = []
    
    for name, param in model.named_parameters():
        if 'head' in name and 'fusion' not in name:
            continue  # 跳过单模态 head
        if 'text' in name:
            record_names_text.append((name, param))
        elif 'audio' in name:
            record_names_audio.append((name, param))
        elif 'visual' in name:
            record_names_visual.append((name, param))
    
    # ==== 初始化 FIM 字典 (用于计算各模态的 Fisher 信息) ====
    fim_text = {}
    fim_audio = {}
    fim_visual = {}
    
    # 获取实际的模型 (处理 DataParallel)
    actual_model = model.module if hasattr(model, 'module') else model
    
    for name, module in actual_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if module.weight.requires_grad:
                # 根据名称分配到不同模态
                if 'text' in name:
                    fim_text[name] = torch.zeros_like(module.weight)
                elif 'audio' in name:
                    fim_audio[name] = torch.zeros_like(module.weight)
                elif 'visual' in name:
                    fim_visual[name] = torch.zeros_like(module.weight)
    
    total_loss = 0
    total_loss_t = 0
    total_loss_a = 0
    total_loss_v = 0
    total_correct = 0
    total_samples = 0
    
    for step, (text, audio, visual, label) in enumerate(dataloader):
        text = text.to(device).float()
        audio = audio.to(device).float()
        visual = visual.to(device).float()
        label = label.to(device).long()
        
        batch_size = label.shape[0]
        total_samples += batch_size
        
        optimizer.zero_grad()
        
        # Forward pass
        t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion = model(text, audio, visual)
        
        # 计算各模态 loss
        loss_fusion = criterion(out_fusion, label)
        loss_t = criterion(out_t, label)
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)
        
        # 计算各模态的 score (softmax 概率在真实标签处的和)
        score_t = sum([softmax(out_t)[i][label[i]] for i in range(batch_size)])
        score_a = sum([softmax(out_a)[i][label[i]] for i in range(batch_size)])
        score_v = sum([softmax(out_v)[i][label[i]] for i in range(batch_size)])
        
        # 计算三模态的 beta
        beta_t, beta_a, beta_v = compute_trimodal_beta(
            score_t, score_a, score_v, alpha=args.beta_scale
        )
        
        # 总 loss
        total_loss_item = loss_fusion + loss_t + loss_a + loss_v
        
        # Backward
        loss_t.backward(retain_graph=True)
        loss_a.backward(retain_graph=True)
        loss_v.backward(retain_graph=True)
        loss_fusion.backward()
        
        # 应用 Proximal 正则化 (根据 beta 调整梯度)
        # 只有当 k > k_threshold 时才应用，表示模型处于 prime learning window
        if args.modulation == 'Ours' and k > args.k_threshold:
            for model_param, global_param in zip(model.parameters(), global_model.parameters()):
                if model_param.requires_grad and model_param.grad is not None:
                    # 检查参数属于哪个模态
                    param_name = None
                    for name, p in model.named_parameters():
                        if p is model_param:
                            param_name = name
                            break
                    
                    if param_name is not None:
                        # 支持两种命名: text_net/audio_net/visual_net 或 text_encoder/audio_encoder/visual_encoder
                        if 'text_net' in param_name or 'text_encoder' in param_name:
                            model_param.grad += beta_t * (model_param - global_param)
                        elif 'audio_net' in param_name or 'audio_encoder' in param_name:
                            model_param.grad += beta_a * (model_param - global_param)
                        elif 'visual_net' in param_name or 'visual_encoder' in param_name:
                            model_param.grad += beta_v * (model_param - global_param)
        
        # ==== 累积 FIM (Fisher Information Matrix) ====
        # FIM ≈ E[∇log p(y|x) * ∇log p(y|x)^T] ≈ 梯度的平方
        for name, module in actual_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if module.weight.requires_grad and module.weight.grad is not None:
                    grad_sq = module.weight.grad.data ** 2
                    if 'text' in name and name in fim_text:
                        fim_text[name] += grad_sq
                    elif 'audio' in name and name in fim_audio:
                        fim_audio[name] += grad_sq
                    elif 'visual' in name and name in fim_visual:
                        fim_visual[name] += grad_sq
        
        optimizer.step()
        scheduler.step()  # Cosine scheduler: step after each batch
        
        # 统计
        total_loss += loss_fusion.item()
        total_loss_t += loss_t.item()
        total_loss_a += loss_a.item()
        total_loss_v += loss_v.item()
        
        _, predicted = torch.max(out_fusion, 1)
        total_correct += (predicted == label).sum().item()
        
        # 打印进度
        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # 判断是否应用了 modulation
            modulation_applied = args.modulation == 'Ours' and k > args.k_threshold
            mod_status = "[MOD ON]" if modulation_applied else "[MOD OFF]"
            print(f"  Step {step}/{len(dataloader)}, "
                  f"Loss: {loss_fusion.item():.4f}, LR: {current_lr:.6f}, "
                  f"beta_t: {beta_t.item():.4f}, beta_a: {beta_a.item():.4f}, beta_v: {beta_v.item():.4f} {mod_status}")
    
    # ==== 计算各模态的 FIM Trace ====
    fim_trace_text = 0
    for name in fim_text:
        fim_text[name] = fim_text[name].mean().item()
        fim_trace_text += fim_text[name]
    
    fim_trace_audio = 0
    for name in fim_audio:
        fim_audio[name] = fim_audio[name].mean().item()
        fim_trace_audio += fim_audio[name]
    
    fim_trace_visual = 0
    for name in fim_visual:
        fim_visual[name] = fim_visual[name].mean().item()
        fim_trace_visual += fim_visual[name]
    
    # 将 FIM trace 添加到列表中 (用于下一个 epoch 计算 k)
    if text_trace_list is not None:
        text_trace_list.append(fim_trace_text)
    if audio_trace_list is not None:
        audio_trace_list.append(fim_trace_audio)
    if visual_trace_list is not None:
        visual_trace_list.append(fim_trace_visual)
    
    print(f"FIM Trace - Text: {fim_trace_text:.6f}, Audio: {fim_trace_audio:.6f}, Visual: {fim_trace_visual:.6f}")
    
    # Note: scheduler.step() 已在每个 batch 后调用 (cosine scheduler)
    
    # 计算平均值
    avg_loss = total_loss / len(dataloader)
    avg_loss_t = total_loss_t / len(dataloader)
    avg_loss_a = total_loss_a / len(dataloader)
    avg_loss_v = total_loss_v / len(dataloader)
    accuracy = total_correct / total_samples
    
    # TensorBoard 记录
    current_lr = optimizer.param_groups[0]['lr']
    if writer is not None:
        writer.add_scalar('Loss/fusion', avg_loss, epoch)
        writer.add_scalar('Loss/text', avg_loss_t, epoch)
        writer.add_scalar('Loss/audio', avg_loss_a, epoch)
        writer.add_scalar('Loss/visual', avg_loss_v, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('FIM_Trace/text', fim_trace_text, epoch)
        writer.add_scalar('FIM_Trace/audio', fim_trace_audio, epoch)
        writer.add_scalar('FIM_Trace/visual', fim_trace_visual, epoch)
        writer.add_scalar('k_value', k, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
    
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, k={k:.6f}")
    
    return avg_loss, avg_loss_t, avg_loss_a, avg_loss_v, accuracy, fim_trace_text, fim_trace_audio, fim_trace_visual, k


def valid_trimodal(args, model, device, dataloader):
    """
    验证/测试函数
    """
    softmax = nn.Softmax(dim=1)
    n_classes = args.n_classes
    
    with torch.no_grad():
        model.eval()
        
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        
        for step, (text, audio, visual, label) in enumerate(dataloader):
            text = text.to(device).float()
            audio = audio.to(device).float()
            visual = visual.to(device).float()
            label = label.to(device).long()
            
            t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion = model(text, audio, visual)
            
            prediction = softmax(out_fusion)
            pred_t = softmax(out_t)
            pred_a = softmax(out_a)
            pred_v = softmax(out_v)
            
            for i in range(text.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                t = np.argmax(pred_t[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                
                lbl = label[i].item()
                num[lbl] += 1.0
                
                if lbl == ma:
                    acc[lbl] += 1.0
                if lbl == t:
                    acc_t[lbl] += 1.0
                if lbl == a:
                    acc_a[lbl] += 1.0
                if lbl == v:
                    acc_v[lbl] += 1.0
    
    total_acc = sum(acc) / sum(num) if sum(num) > 0 else 0
    total_acc_t = sum(acc_t) / sum(num) if sum(num) > 0 else 0
    total_acc_a = sum(acc_a) / sum(num) if sum(num) > 0 else 0
    total_acc_v = sum(acc_v) / sum(num) if sum(num) > 0 else 0
    
    return total_acc, total_acc_t, total_acc_a, total_acc_v


def main():
    args = get_arguments()
    print("=" * 60)
    print("三模态训练 - MOSEI/MOSI")
    print("=" * 60)
    print(args)
    
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, GPU ids: {gpu_ids}")
    
    # 创建模型 - 根据 model_type 选择
    if args.model_type == 'transformer':
        print("Using TriModalTransformer (Text-Transformer, Audio/Visual-MLP)")
        model = TriModalTransformer(args)
    else:
        print("Using TriModalClassifier (Simple MLP for all modalities)")
        model = TriModalClassifier(args)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.to(device)
    
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
    # 数据集 (先加载数据集，因为 scheduler 需要 len(train_loader))
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_mosei_dataloaders(
        args, batch_size=args.batch_size, num_workers=16  # mosei 用 16 workers
    )
    
    # 优化器 (参考 mosei: SGD, lr=0.001, momentum=0.9, weight_decay=1e-4)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                              momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Scheduler (参考 mosei: cosine_schedule_with_warmup)
    # warmup_steps = len(train_loader) * warmup_epochs
    # total_steps = len(train_loader) * n_epochs
    num_warmup_steps = int(len(train_loader) * args.warmup_epochs)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Using CosineScheduleWithWarmup: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
    
    if args.train:
        # TensorBoard
        writer_path = os.path.join(args.tensorboard_path, args.dataset)
        os.makedirs(writer_path, exist_ok=True)
        log_name = f'{args.modulation}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(os.path.join(writer_path, log_name))
        
        best_acc = 0.0
        global_model = cp.deepcopy(model)
        
        # 记录列表
        loss_list = []
        acc_list = []
        acc_t_list = []
        acc_a_list = []
        acc_v_list = []
        
        # ==== FIM Trace 列表 (初始化 10 个 0 做平滑) ====
        # 用于判断 prime learning window
        text_trace_list = [0.0] * 10
        audio_trace_list = [0.0] * 10
        visual_trace_list = [0.0] * 10
        
        # k 值列表 (记录学习窗口状态)
        k_list = []
        
        for epoch in range(args.epochs):
            print(f'\n===== Epoch {epoch} =====')
            
            # 训练
            (avg_loss, loss_t, loss_a, loss_v, train_acc, 
             fim_trace_text, fim_trace_audio, fim_trace_visual, k) = train_epoch_trimodal(
                args, epoch, model, device, dataloader=train_loader,
                optimizer=optimizer, scheduler=scheduler, writer=writer,
                global_model=global_model,
                text_trace_list=text_trace_list,
                audio_trace_list=audio_trace_list,
                visual_trace_list=visual_trace_list
            )
            
            loss_list.append(avg_loss)
            k_list.append(k)
            
            # 验证
            acc, acc_t, acc_a, acc_v = valid_trimodal(args, model, device, val_loader)
            acc_list.append(acc)
            acc_t_list.append(acc_t)
            acc_a_list.append(acc_a)
            acc_v_list.append(acc_v)
            
            writer.add_scalar('Accuracy/val_fusion', acc, epoch)
            writer.add_scalar('Accuracy/val_text', acc_t, epoch)
            writer.add_scalar('Accuracy/val_audio', acc_a, epoch)
            writer.add_scalar('Accuracy/val_visual', acc_v, epoch)
            
            print(f"Val - Fusion: {acc:.4f}, Text: {acc_t:.4f}, Audio: {acc_a:.4f}, Visual: {acc_v:.4f}")
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                os.makedirs(args.ckpt_path, exist_ok=True)
                
                model_name = f'best_model_{args.dataset}_{args.modulation}_epoch{epoch}_acc{acc:.4f}.pth'
                save_path = os.path.join(args.ckpt_path, model_name)
                
                saved_dict = {
                    'epoch': epoch,
                    'modulation': args.modulation,
                    'acc': acc,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(saved_dict, save_path)
                print(f'Best model saved to {save_path}')
            
            # 更新 global_model (每 epoch 更新一次)
            global_model = cp.deepcopy(model)
        
        # 最终测试
        test_acc, test_acc_t, test_acc_a, test_acc_v = valid_trimodal(args, model, device, test_loader)
        print(f"\n===== Final Test Results =====")
        print(f"Fusion: {test_acc:.4f}, Text: {test_acc_t:.4f}, Audio: {test_acc_a:.4f}, Visual: {test_acc_v:.4f}")
        print(f"Best Val Acc: {best_acc:.4f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join('results', f'trimodal_{args.dataset}_{timestamp}')
        os.makedirs(results_path, exist_ok=True)
        
        data_to_save = {
            'loss_list': loss_list,
            'acc_list': acc_list,
            'acc_t_list': acc_t_list,
            'acc_a_list': acc_a_list,
            'acc_v_list': acc_v_list,
            'text_trace_list': text_trace_list,
            'audio_trace_list': audio_trace_list,
            'visual_trace_list': visual_trace_list,
            'k_list': k_list,
        }
        
        for name, data in data_to_save.items():
            pkl_path = os.path.join(results_path, f'{name}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
        
        print(f"Results saved to {results_path}")
        writer.close()
    
    else:
        # 测试模式
        if os.path.exists(args.ckpt_path):
            loaded_dict = torch.load(args.ckpt_path)
            model.load_state_dict(loaded_dict['model'])
            print(f'Model loaded from {args.ckpt_path}')
            
            test_acc, test_acc_t, test_acc_a, test_acc_v = valid_trimodal(args, model, device, test_loader)
            print(f"Test - Fusion: {test_acc:.4f}, Text: {test_acc_t:.4f}, Audio: {test_acc_a:.4f}, Visual: {test_acc_v:.4f}")


if __name__ == "__main__":
    main()

