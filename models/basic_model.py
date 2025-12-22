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


class TriModalClassifier(nn.Module):
    def __init__(self, args):
        super(TriModalClassifier, self).__init__()

        self.dataset = args.dataset
        if args.dataset == 'MOSEI' or args.dataset == 'MOSI':
            n_classes = getattr(args, 'n_classes', 3)
            text_dim = getattr(args, 'text_dim', 300)
            audio_dim = getattr(args, 'audio_dim', 74)
            visual_dim = getattr(args, 'visual_dim', 35)
        else:
            n_classes = getattr(args, 'n_classes', 3)
            text_dim = getattr(args, 'text_dim', 300)
            audio_dim = getattr(args, 'audio_dim', 74)
            visual_dim = getattr(args, 'visual_dim', 35)

        self.n_classes = n_classes
        self.hidden_dim = getattr(args, 'hidden_dim', 512)
        hidden_dim = self.hidden_dim

        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.visual_net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Unimodal Heads (单模态分类头)
        self.head_text = nn.Linear(hidden_dim, n_classes)
        self.head_audio = nn.Linear(hidden_dim, n_classes)
        self.head_visual = nn.Linear(hidden_dim, n_classes)
        
        # Fusion Head (多模态融合分类头)
        self.fusion_head = nn.Linear(hidden_dim * 3, n_classes)
        
        # 用于 Proximal 正则化的标志
        self.usegate = False

    def forward(self, text, audio, visual):
        """
        Forward pass
        
        Args:
            text: [B, T, D_text] 或 [B, D_text] 文本特征
            audio: [B, T, D_audio] 或 [B, D_audio] 音频特征
            visual: [B, T, D_visual] 或 [B, D_visual] 视觉特征
        
        Returns:
            t_feat: [B, hidden_dim] 文本特征
            a_feat: [B, hidden_dim] 音频特征
            v_feat: [B, hidden_dim] 视觉特征
            out_t: [B, n_classes] 文本单模态输出
            out_a: [B, n_classes] 音频单模态输出
            out_v: [B, n_classes] 视觉单模态输出
            out_fusion: [B, n_classes] 融合输出
        """
        # 处理输入维度，如果是序列 [B, T, D] 则取均值池化
        if text.dim() == 3: 
            text = text.mean(dim=1)
        if audio.dim() == 3: 
            audio = audio.mean(dim=1)
        if visual.dim() == 3: 
            visual = visual.mean(dim=1)

        # 提取特征
        t_feat = self.text_net(text)
        a_feat = self.audio_net(audio)
        v_feat = self.visual_net(visual)

        # 单模态输出
        out_t = self.head_text(t_feat)
        out_a = self.head_audio(a_feat)
        out_v = self.head_visual(v_feat)

        # 融合输出
        combined = torch.cat((t_feat, a_feat, v_feat), dim=1)
        out_fusion = self.fusion_head(combined)

        return t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion
    
    def get_fusion_output_from_features(self, t_feat, a_feat, v_feat):
        """从已有特征计算融合输出 (用于分析)"""
        combined = torch.cat((t_feat, a_feat, v_feat), dim=1)
        return self.fusion_head(combined)


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
