#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .attention import CA_Block, SSH
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from nets.backbone import Backbone, Block, Conv, SiLU, Transition, autopad


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [128,256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    #新增一个浅层特征层
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark2", "dark3", "dark4", "dark5"), in_channels = [128, 256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(3 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        #four scale module
        #80x80x256->80x80x128
        self.reduce_conv2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #160x160x1024->160x160x128
        self.reduce_conv1024 = BaseConv(int(in_channels[3] * width), int(in_channels[0] * width), 1, 1, act=act)
        #160x160x1024->80x80x256
        self.reduce_conv256 = BaseConv(int(in_channels[3] * width), int(in_channels[1] * width), 1, 1, act=act)
        #80x80x512->80x80x128
        self.reduce_conv512 = BaseConv(int(in_channels[2] * width), int(in_channels[0] * width), 1, 1, act=act)
        #160x160x256->160x160x128
        self.C2 = CSPLayer(
            int(4 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        #   80, 80, 256 -> 80, 80, 256
        self.C2_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        #   160, 160, 128 -> 80, 80, 128
        self.bu_conv3 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #define CA attention
        self.feat0_attention=CA_Block(64)
        self.feat1_attention=CA_Block(128)
        #define SSH module
        self.ssh1=SSH(64,64)
        self.ssh2=SSH(128,128)
        # self.ssh3=SSH(128,128)

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat0, feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #feat0 add CA_attention
        feat0=self.feat0_attention(feat0)

        #feat1 add CA_attention
        feat1=self.feat1_attention(feat1)

        #跳跃连接1 --- 20x20x1024->160x160x1024
        p5_up1=self.upsample(self.upsample(self.upsample(feat3)))
        #reduce_conv
        p5_up1=self.reduce_conv1024(p5_up1)

        # 跳跃连接2 --- 20x20x1024->80x80x1024
        p5_up2 = self.upsample(self.upsample(feat3))
        p5_up2=self.reduce_conv256(p5_up2)
        # 跳跃链接3 --- 40x40x512->160x160x512
        p4_up = self.upsample(self.upsample(feat2))
        p4_up=self.reduce_conv512(p4_up)

        #160x160x(128+128+128)
        feat0=torch.cat([feat0,p5_up1,p4_up],1)

        #80x80x(256+256)
        feat1=torch.cat([feat1,p5_up2],1)

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)

        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out0      = self.C3_p3(P4_upsample)

        #Add P2
        # P3_out sampling,P3缩小通道数后上采样
        # 80x80x256->80x80x128
        P2 = self.reduce_conv2(P3_out0)
        # 80x80x128->160x160x128
        P3_upsample = self.upsample(P2)
        # concat后csp 160x160x512
        p2_cat = torch.cat([P3_upsample, feat0], 1)
        #P2_out  ------160x160x512->160x160x128
        P2_out = self.C2(p2_cat)


        #160x160x128->80x80x128
        P2_downsample=self.bu_conv3(P2_out)
        #80x80x256->80x80x128
        P3=self.reduce_conv2(P3_out0)

        #P3_out  80x80x256->80x80x256
        P3_cat=torch.cat([ P2_downsample,P3],1)
        P3_out=self.C2_n3(P3_cat)

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out)
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        #add SSH
        P2_out=self.ssh1(P2_out)
        P3_out=self.ssh2(P3_out)

        return (P2_out,P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
