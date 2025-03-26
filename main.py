import argparse
import os
import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import torch.nn.functional as F
from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
import torchvision
import os
import pickle
import csv
from datetime import datetime
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str)

    parser.add_argument('--fusion_method', default='concat', type=str)
    parser.add_argument('--fps', default=3, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/home/hcx/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/home/hcx/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=40, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    
    parser.add_argument('--ckpt_path', default= r'/mnt/sda/data_chengxiang/mm_model', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='/home/hcx/InfoReg/logs',type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='GPU ids')
    parser.add_argument('--audio_fim_path',default='/home/hcx/audio_fim_folder',type=str,help='path to store the audio fim')
    parser.add_argument('--visual_fim_path',default='/home/hcx/visual_fim_folder',type=str,help='path to store the visual fim')
    parser.add_argument('--mm_fim_path',default='/home/hcx/mm_fim_folder',type=str,help='path to store the mm fim')
    parser.add_argument('--accuracy_path',default='/home/hcx/accuracy_folder',type=str,help='path to store accuracy csv file')

    return parser.parse_args()




def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, visual_trace_list=None, audio_trace_list=None):
    total_audio_grad_sum = 0
    total_visual_grad_sum = 0
    total_audio_count = 0
    total_visual_count = 0

    if epoch == 0:
        k = 1
    else:
        tr1 = sum(audio_trace_list[-10:])/10
        tr2 = sum(audio_trace_list[-11:-1])/10
        
        # tr1 = audio_trace_list[-1]
        # tr2 = audio_trace_list[-2]
        k = (tr1 - tr2)/tr1
            
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    global_model = cp.deepcopy(model)
    model.train()
    print("Start training ... ")

    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name: 
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    
    fim = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if module.weight.requires_grad:
                fim[name] = torch.zeros_like(module.weight)
    fim_audio = {}
    fim_visual = {}
    fim_audio_head = {}
    fim_visual_head = {}
    fim_mm_head = {}

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if 'audio_net' in name:
                if module.weight.requires_grad:
                    fim_audio[name] = torch.zeros_like(module.weight)
            elif 'visual_net' in name:
                if module.weight.requires_grad:
                    fim_visual[name] = torch.zeros_like(module.weight)
            elif 'head_audio' in name:
                if module.weight.requires_grad:
                    fim_audio_head[name] = torch.zeros_like(module.weight)
            elif 'head_video' in name:
                if module.weight.requires_grad:
                    fim_visual_head[name] = torch.zeros_like(module.weight)
            elif 'head' in name:
                if module.weight.requires_grad:
                    fim_mm_head[name] = torch.zeros_like(module.weight)
            


    for step, (spec, image, label) in enumerate(dataloader):

        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        lr = optimizer.param_groups[0]['lr']
        a,v,out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())

        # if args.fusion_method == 'sum':
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
        #              model.module.fusion_module.fc_y.bias)
        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
        #              model.module.fusion_module.fc_x.bias)
        # else:
        #     weight_size = model.module.fusion_module.fc_out.weight.size(1)
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        loss = criterion(out, label)
        loss_out_v = criterion(out_v, label)
        loss_out_a = criterion(out_a, label)


        losses=[loss,loss_out_a,loss_out_v]
        all_loss = ['both', 'audio', 'visual']
        grads_audio = {}
        grads_visual={}

        for idx, loss_type in enumerate(all_loss):
            loss_tem = losses[idx]
            loss_tem.backward(retain_graph=True)
            if(loss_type=='visual'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_visual[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_visual])           
                average_grad_visual = torch.mean(grads_visual[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_visual',average_grad_visual,epoch)
            elif(loss_type=='audio'):
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    if param.grad is not None:
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_audio[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_audio])
                average_grad_audio = torch.mean(grads_audio[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_audio',average_grad_audio,epoch)
            else:
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    if param.grad is not None:
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_audio[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                average_grad_audio_mm = torch.mean(grads_audio[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_audio_mm',average_grad_audio_mm,epoch)
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    else:
                        grads_visual[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])
                average_grad_visual_mm = torch.mean(grads_visual[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_visual_mm',average_grad_visual_mm,epoch)
            
            optimizer.zero_grad()


        loss_out_v.backward(retain_graph=True)  
        loss_out_a.backward(retain_graph=True)  
        loss.backward()
        
        score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
        score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
        
        
        if(score_a > score_v) and k > 0.04:
            gap = score_a - score_v
            gap = tanh(score_a - score_v)
            beta = 0.95 * torch.exp(gap)
            beta2 = 0
        elif(score_v < score_a) and k > 0.04:
            gap = score_v - score_a
            gap = tanh(score_v - score_a)
            beta2 = 0.1 * torch.exp(gap)
            beta = 0
        else:
            beta = 0
            beta2 = 0

            
        for model_param, global_model_param in zip(model.parameters(),global_model.parameters()):
               
            if model_param.requires_grad and any(model_param is param for param in model.module.audio_net.parameters()):
                model_param.grad += beta * (model_param - global_model_param)

            if model_param.requires_grad and any(model_param is param for param in model.module.visual_net.parameters()): 
                model_param.grad += beta2 * (model_param - global_model_param)
                    
        

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                if module.weight.requires_grad:
                    fim[name] += (module.weight.grad * module.weight.grad)
                    fim[name].detach_()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if 'audio_net' in name and module.weight.requires_grad:
                    fim_audio[name] += (module.weight.grad * module.weight.grad)
                    fim_audio[name].detach_()
                elif 'visual_net' in name and module.weight.requires_grad:
                    fim_visual[name] += (module.weight.grad * module.weight.grad)
                    fim_visual[name].detach_()
                elif 'head_audio' in name and module.weight.requires_grad:
                    fim_audio_head[name] += (module.weight.grad * module.weight.grad)
                    fim_audio_head[name].detach_()
                elif 'head_video' in name and module.weight.requires_grad:
                    fim_visual_head[name] += (module.weight.grad * module.weight.grad)
                    fim_visual_head[name].detach_()
                elif 'head' in name and module.weight.requires_grad:
                    fim_mm_head[name] += (module.weight.grad * module.weight.grad)
                    fim_mm_head[name].detach_()

        for name, parms in model.named_parameters():
            if parms.grad is not None: 
                layer = str(name).split('.')[1]

                if 'audio' in layer and len(parms.grad.size()) == 4:

                    audio_L2_norm_square = torch.sum(parms.grad ** 2).item()
                    total_audio_grad_sum += lr * audio_L2_norm_square
                    total_audio_count += 1


                if 'visual' in layer and len(parms.grad.size()) == 4:

                    visual_L2_norm_square = torch.sum(parms.grad ** 2).item()
                    total_visual_grad_sum += lr * visual_L2_norm_square
                    total_visual_count += 1


        optimizer.step()

      

        

        _loss += loss.item()
        _loss_a += loss_out_a.item()
        _loss_v += loss_out_v.item()

    if total_audio_count > 0:
        epoch_audio_L2_norm_mean = total_audio_grad_sum / total_audio_count
    else:
        epoch_audio_L2_norm_mean = 0

    if total_visual_count > 0:
        epoch_visual_L2_norm_mean = total_visual_grad_sum / total_visual_count
    else:
        epoch_visual_L2_norm_mean = 0


    fim_trace = 0
    for name in fim:
        fim[name] = fim[name].mean().item()
        fim_trace += fim[name]

    fim_trace_audio = 0
    for name in fim_audio:
        fim_audio[name] = fim_audio[name].mean().item()
        fim_trace_audio += fim_audio[name]

    fim_trace_visual = 0
    for name in fim_visual:
        fim_visual[name] = fim_visual[name].mean().item()
        fim_trace_visual += fim_visual[name]

    fim_trace_visual_head = 0
    for name in fim_visual_head:
        fim_visual_head[name] = fim_visual_head[name].mean().item()
        fim_trace_visual_head += fim_visual_head[name]

    fim_trace_audio_head = 0
    for name in fim_audio_head:
        fim_audio_head[name] = fim_audio_head[name].mean().item()
        fim_trace_audio_head += fim_audio_head[name]

    fim_trace_mm_head = 0
    for name in fim_mm_head:
        fim_mm_head[name] = fim_mm_head[name].mean().item()
        fim_trace_mm_head += fim_mm_head[name]
    
    visual_trace_list.append(fim_trace_visual)
    audio_trace_list.append(fim_trace_audio)

    writer.add_scalar('fim_trace_visual_head',fim_trace_visual_head,epoch)
    writer.add_scalar('fim_trace_audio_head',fim_trace_audio_head,epoch)
    writer.add_scalar('fim_trace_mm_head',fim_trace_mm_head,epoch)

    scheduler.step()


    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual, average_grad_visual, average_grad_audio, average_grad_visual_mm, average_grad_audio_mm


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v,a_output,v_output, out = model(spec.unsqueeze(1).float(), image.float())

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                         model.module.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                         model.module.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)
            pred_v = softmax(v_output)
            pred_a = softmax(a_output)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)

    audio_NormList = [0,0,0,0,0,0,0,0,0,0]
    visual_NormList = [0,0,0,0,0,0,0,0,0,0]
    audio_FGNList = []
    visual_FGNList = []
    audio_GNorm_list = []
    visual_GNorm_list = []
    fim_list = []
    audio_fim_list = []
    visual_fim_list = []
    accuracy_list = []
    accuracy_list_audio = []
    accuracy_list_visual = []
    avarage_gradient_visual_mm_list = []
    avarage_gradient_audio_mm_list = []
    avarage_gradient_visual_list = []
    avarage_gradient_audio_list = []
    
    visual_trace_list = [0,0,0,0,0,0,0,0,0,0]
    audio_trace_list = [0,0,0,0,0,0,0,0,0,0]



    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')

    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    print(f"Number of samples in training dataset: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)
    # writer_path = os.path.join(args.tensorboard_path, args.dataset)
    # if not os.path.exists(writer_path):
    #     os.mkdir(writer_path)
    # log_name = '{}_{}'.format(args.fusion_method, args.modulation)
    # writer = SummaryWriter(os.path.join(writer_path, log_name))
    # log_images_to_tensorboard(writer, train_dataloader, device, phase='train')

    if args.train:

        best_acc = 0.0

        for epoch in range(args.epochs):
            total_audio_grad_sum = 0
            total_visual_grad_sum = 0
            total_audio_count = 0
            total_visual_count = 0

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))
                
                
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace ,fim_trace_audio,fim_trace_visual,average_grad_visual,average_grad_audio,average_grad_visual_mm,average_grad_audio_mm = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,writer,visual_trace_list,audio_trace_list)
                fim_list.append(fim_trace)
                avarage_gradient_visual_list.append(average_grad_visual)
                avarage_gradient_audio_list.append(average_grad_audio)
                avarage_gradient_visual_mm_list.append(average_grad_visual_mm)
                avarage_gradient_audio_mm_list.append(average_grad_audio_mm)
                audio_fim_list.append(fim_trace_audio)
                visual_fim_list.append(fim_trace_visual)
                # visual_trace_list.append(fim_trace_visual)
                # audio_trace_list.append(fim_trace_audio)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                accuracy_list.append(acc)
                accuracy_list_visual.append(acc_v)
                accuracy_list_audio.append(acc_a)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-11:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-11:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)

                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-11:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-11:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)


                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

                writer.add_scalar('FIM_Trace', fim_trace,epoch)
                writer.add_scalar('FIM_Trace_audio', fim_trace_audio,epoch)
                writer.add_scalar('Fim_Trace_visual',fim_trace_visual,epoch)
                                            

            else:
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-2:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-2:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)
                print("audio_FGN:", (audio_NewNorm - audio_OldNorm) / audio_OldNorm)
                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-2:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-2:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)
                print("visual_FGN:", (visual_NewNorm - visual_OldNorm) / visual_OldNorm)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
        print("AUDIO_FGN:",audio_FGNList)
        print("Visual_FGN:",visual_FGNList)
        print("audio梯度:",audio_GNorm_list)
        print("visual梯度:",visual_GNorm_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"experiments_{timestamp}"

        results_path = os.path.join(os.getcwd(), "results", experiment_folder)
        os.makedirs(results_path, exist_ok=True)


        data_to_save = {
            "fim_list": fim_list,
            "audio_fim_list": audio_fim_list,
            "visual_fim_list": visual_fim_list,
            "accuracy_list": accuracy_list,
            "accuracy_list_audio": accuracy_list_audio,
            "accuracy_list_visual": accuracy_list_visual,
            "avarage_gradient_visual_mm_list": avarage_gradient_visual_mm_list,
            "avarage_gradient_audio_mm_list": avarage_gradient_audio_mm_list,
            "avarage_gradient_visual_list": avarage_gradient_visual_list,
            "avarage_gradient_audio_list": avarage_gradient_audio_list
            
        }


        for name, data in data_to_save.items():

            pkl_path = os.path.join(results_path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)


            csv_path = os.path.join(results_path, f"{name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)

                if isinstance(data[0], (list, tuple)):
                    writer.writerows(data)
                else:
                    writer.writerow(data)


    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
