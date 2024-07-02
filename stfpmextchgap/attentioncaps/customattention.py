import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from .p_c_att import PAM_Module, CAM_Module
# from p_c_att import PAM_Module, CAM_Module

  
class CHGAP(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.ch = CAM_Module(c)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.ecaconv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) # k= 3

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, _BatchNorm):
        #         m.weight.data.fill_(1)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        self.ecasigmoid = nn.Sigmoid()


    def forward(self, x):
        ### Channel and ECA ####
        channelattention = self.ch(x)
        # print(f'channelattention: {channelattention.shape}')
        gap = self.gap(channelattention)
        # print(f'gap: {gap.shape}')
        ecaconv = self.ecaconv(gap.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print(f'ecaconv: {ecaconv.shape}')
        ## channel and eca end ###
        ecasigh = self.ecasigmoid(ecaconv)
        ecaout = channelattention * ecasigh
        return ecaout




class EXTAttention(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()

        self.query_conv = nn.Conv2d(c, c, kernel_size=1)

        self.conv1d_k = nn.Conv1d(c, c, kernel_size=1, bias=False)
        self.conv1d_v = nn.Conv1d(c, c, kernel_size=1, bias=False)
        self.conv1d_v.weight.data = self.conv1d_k.weight.data.permute(1, 0, 2)

        self.outconv2dbn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )        

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, _BatchNorm):
        #         m.weight.data.fill_(1)
        #         if m.bias is not None:
        #             m.bias.data.zero_()



    def forward(self, x):
        ### Take the input ###
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C, width*height)
        # print(f'Query: {proj_query.shape}')
        attn_mk = self.conv1d_k(proj_query)
        # print(f'attn_mk: {attn_mk.shape}')
        ### Normalization ###
        attn_mk_sf = self.tempered_softmax(x=attn_mk,temperature=2.0)
        # print(f'attn_mk_sf: {attn_mk_sf.shape}')
        ### End Normalization ###
        attn_mv = self.conv1d_v(attn_mk_sf)
        # print(f'attn_mv: {attn_mv.shape}')
        attn_reshape = attn_mv.view(m_batchsize, C, height, width)
        # print(f'reshape: {attn_reshape.shape}')     
        outconv = self.outconv2dbn(attn_reshape)
        # print(f'outconv: {outconv.shape}')
        ### end EXternal ####


        return outconv
    
    
    def tempered_softmax(self, x, temperature=1.0):
        x_temp = x / temperature
        return F.softmax(x_temp, dim=-1)
    


class EXTCHGAP(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        #### External Attention
        self.convex = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()            
        )
        self.extattention = EXTAttention(in_channel)

        self.conv1ex = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()            
        )

        self.outconvex = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )
        ### Channel Attention
        self.convch = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()            
        )
        self.chgap = CHGAP(in_channel)      
                
        self.ecasigmoid = nn.Sigmoid()        


    def forward(self, x):
        ### extattention
        exfeatures = self.convex(x)
        # print(f'exfeatures: {exfeatures.shape}')
        extattention = self.extattention(exfeatures)
        # print(f'extattention: {extattention.shape}')
        ex_conv = self.conv1ex(extattention)
        # print(f'ex_conv: {ex_conv.shape}')
        exoutfeatures = self.outconvex(ex_conv)
        # print(f'exoutfeatures: {exoutfeatures.shape}')        
              
        ### channel and ECA ####
        chfeatures = self.convch(x)
        # print(f'chfeatures: {chfeatures.shape}')
        chgap = self.chgap(chfeatures)
        # print(f'chgap: {chgap.shape}')
        

        # fusion = exoutfeatures * chgap
        fusion = exoutfeatures + chgap
        # print(f'fusion: {fusion.shape}')

        # return self.ecasigmoid(fusion)
        return fusion


if __name__ == "__main__":
    # Test model
    torch.manual_seed(42)
    # model = CHGAP(256)
    model = EXTAttention(256)
    # model = EXTCHGAP(256, 256)

    input_tensor = torch.randn(1, 256, 28, 28)
    output = model(input_tensor)



    print(output.shape)
    # print(output)