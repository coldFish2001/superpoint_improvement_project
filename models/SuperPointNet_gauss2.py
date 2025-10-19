"""latest version of SuperpointNet. Use it!

"""

#代码总结：
#1.构建 SuperPoint 神经网络的骨架（编码器和双头部）。
#2.实现特征点应用任务，即使用描述符进行鲁棒的双向匹配。
#3.main:测试和验证模型的功能（初始化、前向传播、后处理）和性能（计时）。

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np

# from models.SubpixelNet import SubpixelNet
class SuperPointNet_gauss2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
        super(SuperPointNet_gauss2, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256 #输出通道
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4) #输出最终共享特征图 128维
        # self.down4 = down(c4, 512)
        # self.up1 = up(c4+c3, c2)
        # self.up2 = up(c2+c2, c1)
        # self.up3 = up(c1+c1, c1)
        # self.outc = outconv(c1, subpixel_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)

        # Detector Head.兴趣点检测器头部 128->256->65
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1) #共享特征图延迟到这一步才升维成256
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.描述符头部
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None



    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa)) #semi = $$\mathbf{(N, 65, H/8, W/8)}$$
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))#W/8*H/8*256
        # print("desc: ", desc.shape)

        dn = torch.norm(desc, p=2, dim=1) # dn 的形状是 $(N, 1, H_c, W_c)$ 它保存了H/8*W/8个特征向量的L2-norm,特征向量为256维
        desc = desc.div(torch.unsqueeze(dn, 1)) # L2范数归一化  desc=$(N, 256, H_c, W_c)$
        output = {'semi': semi, 'desc': desc}
        self.output = output

        return output

    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten 65维->1维
        heatmap = flattenDetection(semi) # softmax+reshape = [batch_size, 1, H, W] :全分辨率（W*H）的兴趣点热力图
        # nms 非极大值抑制 消除由卷积和 SoftMax 产生的多余的、邻近的冗余检测。
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets实现亚像素级别的精确兴趣点定位
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred'] #亚像素偏移量

        # extract points 最终outs包含：精确的亚像素坐标+提取出的描述符向量
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output  
        #output 字典现在包含 'semi'（兴趣点原始分数）, 'desc'（半密集描述符张量）, 'keypoints'（最终的兴趣点坐标列表）, 'descriptors'（最终的描述符向量矩阵）


def get_matches(deses_SP): #利用 最近邻（Nearest Neighbor, NN）搜索 来计算两组 SuperPoint 描述符 之间的双向匹配关系
    #deses_SP 就是两张图像各自的、经过 L2 归一化的 256 维 SuperPoint 描述符的集合
    from models.model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    # tracker = PointTracker(max_length=2, nn_thresh=1.2)
    # print("deses_SP[1]: ", deses_SP[1].shape)
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).T, f(deses_SP[1]).T, nn_thresh=1.2)
    return matching_mask #返回一个匹配掩码

    # print("matching_mask: ", matching_mask.shape)
    # f_mask = lambda pts, maks: pts[]
    # pts_m = []
    # pts_m_res = []
    # for i in range(2):
    #     idx = xs_SP[i][matching_mask[i, :].astype(int), :]
    #     res = reses_SP[i][matching_mask[i, :].astype(int), :]
    #     print("idx: ", idx.shape)
    #     print("res: ", idx.shape)
    #     pts_m.append(idx)
    #     pts_m_res.append(res)
    #     pass

    # pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
    # matches_test = toNumpy(pts_m)
    # print("pts_m: ", pts_m.shape)

    # pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
    # # pts_m_res = toNumpy(pts_m_res)
    # print("pts_m_res: ", pts_m_res.shape)
    # # print("pts_m_res: ", pts_m_res)
        
    # pts_idx_res = torch.cat((pts_m, pts_m_res), dim=1)
    # print("pts_idx_res: ", pts_idx_res.shape)

def main():
    #！！！这段代码（main 函数的初始化和测试部分）执行的仅仅是模型的功能验证和推理（Inference）工作，而没有对模型进行任何训练。  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet_gauss2() #实例化模型
    model = model.to(device) #设备【分配

 
    # check keras-like model summary using torchsummary  模型结构检查
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

    ## test 原始输出测试
    image = torch.zeros((2,1,120, 160))
    outs = model(image.to(device))
    print("outs: ", list(outs))

    # 后处理初始化
    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')

    from models.model_utils import SuperPointNet_process 
    params = {
        'out_num_points': 500, #输出点数量限制
        'patch_size': 5, #Soft-Argmax 窗口大小
        'device': device,
        'nms_dist': 4, #NMS 距离（半径）
        'conf_thresh': 0.015 #检测置信度阈值。 原始热力图上的最小激活值。只有像素值高于此阈值，才有资格被考虑为兴趣点
    }

    sp_processer = SuperPointNet_process(**params) #实例化sp后期处理类
    outs = model.process_output(sp_processer) #调用sp_processer类中的方法，实现后处理

    print("outs: ", list(outs))
    print_dict_attr(outs, 'shape')


    #性能基准测试，无需改动
    # timer
    import time
    from tqdm import tqdm
    iter_max = 50

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
    end = time.time()
    print("forward only: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    xs_SP, deses_SP, reses_SP = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
        outs = model.process_output(sp_processer)
        xs_SP.append(outs['pts_int'].squeeze())
        deses_SP.append(outs['pts_desc'].squeeze())
        reses_SP.append(outs['pts_offset'].squeeze())
    end = time.time()
    print("forward + process output: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(len(xs_SP))):
        get_matches([deses_SP[i][0], deses_SP[i][1]])
    end = time.time()
    print("nn matches: ", iter_max/(end - start), " iters/s")


if __name__ == '__main__':
    main()



