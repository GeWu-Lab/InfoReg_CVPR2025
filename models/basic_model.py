import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
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

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        


        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.head = nn.Linear(1024, n_classes)
        self.head2 = nn.Linear(512, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)
        self.fc_x = nn.Linear(512,512)
        self.fc_y = nn.Linear(512,512)
        self.sigmoid = nn.Sigmoid()
        self.usegate = False

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        if self.usegate:
            out_a = self.fc_x(a)
            out_v = self.fc_y(v)
            out_audio=self.head_audio(a)
            out_video=self.head_video(v)
            gate = self.sigmoid(out_a)
            out = self.head2(torch.mul(gate, out_v))
            
        else:
            out = torch.cat((a,v),1)
            out = self.head(out)

            out_audio=self.head_audio(a)
            out_video=self.head_video(v)
            


        return a,v,out_audio,out_video,out


# class AVClassifier_transformer(nn.Module):
#     def __init__(self, args):
#         super(AVClassifier_transformer, self).__init__()

#         fusion = args.fusion_method
#         if args.dataset == 'VGGSound':
#             n_classes = 309
#         elif args.dataset == 'KineticSound':
#             n_classes = 31
#         elif args.dataset == 'CREMAD':
#             n_classes = 6
#         elif args.dataset == 'AVE':
#             n_classes = 28
#         else:
#             raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

#         if fusion == 'sum':
#             self.fusion_module = SumFusion(output_dim=n_classes)
#         elif fusion == 'concat':
#             self.fusion_module = ConcatFusion(output_dim=n_classes)
#         elif fusion == 'film':
#             self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
#         elif fusion == 'gated':
#             self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
#         else:
#             raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        


#         self.audio_net = resnet18(modality='audio')
#         self.visual_net = resnet18(modality='visual')
#         self.head = nn.Linear(1024, n_classes)
#         # 加载预训练的 Transformer
#         self.transformer = MultiModalTransformer(num_classes=n_classes, nframes=3, multi_depth=0, depth=4)
#         loaded_dict = torch.load('/home/chengxiang_huang/unified_framework_mm/pretrained/multi2_vit_pretrain_4s.pth')
#         self.transformer.load_state_dict(loaded_dict, strict=False)
#         self.head_audio = nn.Linear(512, n_classes)
#         self.head_video = nn.Linear(512, n_classes)

#     def forward(self, audio, visual):

#         a = self.audio_net(audio)
#         v = self.visual_net(visual)

#         (_, C, H, W) = v.size()
#         B = a.size()[0]
#         v = v.view(B, -1, C, H, W)
#         v = v.permute(0, 2, 1, 3, 4)

#         a = F.adaptive_avg_pool2d(a, 1)
#         v = F.adaptive_avg_pool3d(v, 1)

#         a = torch.flatten(a, 1)
#         v = torch.flatten(v, 1)

#         out = torch.cat((a,v),1)
#         out = self.transformer(combined_features)

#         out_audio=self.head_audio(a)
#         out_video=self.head_video(v)


#         return a,v,out_audio,out_video,out
