
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactSSMLiteBlock(nn.Module):
    """
    【轻量高效版】精简多方向特征交互块
    剔除大卷积带来的过拟合风险，采用轻量级核（3x3 与 5x5）精准锁定细胞核边缘。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel_proj = nn.Conv2d(channels, channels * 2, 1, bias=False)
        
        # 适度缩减感受野，保留细胞核边缘和局部分离度
        self.scan_h = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=False)
        self.scan_v = nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=False)
        self.scan_d = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        
        
        self.fusion = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU(inplace=True)
        
        # 轻量级通道注意力
        n = 8
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // n, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // n, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_proj = self.channel_proj(x)
        x1, x2 = torch.chunk(x_proj, 2, dim=1)
        
        h = self.scan_h(x1)
        v = self.scan_v(x1)
        d = self.scan_d(x1)
        
        scanned = self.fusion(torch.cat([h, v, d], dim=1))
        scanned = scanned * self.gate(scanned)
        
        out = self.act(self.norm(scanned + x2))
        return shortcut + out


class SOTAPathologyDetectorLite(nn.Module):
    """
    轻量高效版病理细胞核检测头（兼顾泛化性与精度）
    """
    def __init__(self, in_channels_list, feat_channels=96, num_classes=2, space=8):
        super().__init__()
        self.space = space
        self.num_scales = len(in_channels_list)
        
        # 1. 特征投影与对齐
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, feat_channels, 1, bias=False),
                nn.GroupNorm(8, feat_channels),
                nn.SiLU(inplace=True)
            ) for c in in_channels_list
        ])
        
        # 2. 单层高效金字塔融合
        self.pyramid_fuse = nn.Sequential(
            nn.Conv2d(feat_channels * self.num_scales, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3. 轻量级块，防止过拟合
        self.body_blocks = nn.Sequential(
            CompactSSMLiteBlock(feat_channels),
            CompactSSMLiteBlock(feat_channels),
            CompactSSMLiteBlock(feat_channels),
            #CompactSSMLiteBlock(feat_channels)
        )
        
        # 4. 直接输出分类与回归（去除多余的质量掩码，让梯度直接流向坐标）
        self.cls_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feat_channels, num_classes, 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feat_channels, 2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # 回归头小方差初始化
        nn.init.normal_(self.reg_head[-1].weight, std=0.001)
        if self.reg_head[-1].bias is not None:
            nn.init.constant_(self.reg_head[-1].bias, 0)
            
        # 分类头 Focal Loss 先验初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_head[-1].weight, std=0.01)
        if self.cls_head[-1].bias is not None:
            nn.init.constant_(self.cls_head[-1].bias, bias_value)

    def _generate_anchors(self, images, feat_h, feat_w):
        bs, _, h, w = images.shape
        dev, dt = images.device, images.dtype

        grid_y, grid_x = torch.meshgrid(
            torch.arange(feat_h, device=dev, dtype=dt),
            torch.arange(feat_w, device=dev, dtype=dt),
            indexing="ij",
        )
        anchors = torch.stack([grid_x, grid_y], dim=-1) * self.space
        origin_coord = (
            torch.tensor(
                [w % self.space or self.space, h % self.space or self.space],
                device=dev,
                dtype=dt,
            )
            / 2.0
        )
        anchors = anchors + origin_coord
        return anchors.unsqueeze(0).repeat(bs, 1, 1, 1)

    def forward(self, features, images):
        target_size = features[0].shape[2:]
        
        # 多尺度对齐与融合
        aligned = [proj(f) if f.shape[2:] == target_size else F.interpolate(proj(f), size=target_size, mode="bilinear", align_corners=False) for proj, f in zip(self.projs, features)]
        fused = self.pyramid_fuse(torch.cat(aligned, dim=1))
        
        # 轻量级状态空间特征演化
        x_feat = self.body_blocks(fused)
        #x_feat = fused
        feat_h, feat_w = x_feat.shape[2:]
        
        # 生成网格锚点
        anchors = self._generate_anchors(images, feat_h, feat_w)
        
        # 直达式解耦预测
        cls_out = self.cls_head(x_feat).permute(0, 2, 3, 1).contiguous()
        reg_out = self.reg_head(x_feat).permute(0, 2, 3, 1).contiguous()
        
        # 坐标解码
        reg_offset = (torch.sigmoid(reg_out) - 0.5) * (2 * self.space)
        pred_coords = anchors + reg_offset
        
        return pred_coords, cls_out


# =====================================================================
# 终极包装入口 NucScanDet (严格锁定输出格式与形状)
# =====================================================================
class NucScanDet(nn.Module):
    def __init__(self, backbone_cfg=None, fpn_cfg=None, head_cfg=None):
        super().__init__()
        fpn_cfg = fpn_cfg or {}
        head_cfg = head_cfg or {}

        in_channels = fpn_cfg.get("in_channels", [48, 96, 192])
        feat_channels = head_cfg.get("feat_channels", 96)
        num_classes = head_cfg.get("num_classes", 2)
        space = head_cfg.get("space", 8)

        self.backbone = build_backbone(backbone_cfg) if 'build_backbone' in globals() else nn.Identity()

        self.detector = SOTAPathologyDetectorLite(
            in_channels_list=in_channels,
            feat_channels=feat_channels,
            num_classes=num_classes,
            space=space,
        )

    def forward(self, x):
        features = self.backbone(x) if hasattr(self.backbone, '__call__') else [x, x, x]
        pred_coords, pred_logits = self.detector(features, x)

        return {
            "pred_coords": pred_coords.flatten(1, 2).contiguous(),
            "pred_logits": pred_logits.flatten(1, 2).contiguous()
        }

