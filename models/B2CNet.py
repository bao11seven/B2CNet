
import torch
import torch.nn as nn

from models.backbone import resnet18


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)




class BFAM(nn.Module):
    def __init__(self,inp,out):
        super(BFAM, self).__init__()

        self.pre_siam = simam_module()
        self.lat_siam = simam_module()


        out_1 = int(inp/2)

        self.conv_1 = nn.Conv2d(inp, out_1 , padding=1, kernel_size=3,groups=out_1,
                                   dilation=1)
        self.conv_2 = nn.Conv2d(inp, out_1, padding=2, kernel_size=3,groups=out_1,
                                   dilation=2)
        self.conv_3 = nn.Conv2d(inp, out_1, padding=3, kernel_size=3,groups=out_1,
                                   dilation=3)
        self.conv_4 = nn.Conv2d(inp, out_1, padding=4, kernel_size=3,groups=out_1,
                                   dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )

        self.fuse_siam = simam_module()

        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self,inp1,inp2,last_feature=None):
        x = torch.cat([inp1,inp2],dim=1)
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1,c2,c3,c4],dim=1)
        fuse = self.fuse(cat)
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)


        inp1_mul = torch.mul(inp1_siam,fuse)
        inp2_mul = torch.mul(inp2_siam,fuse)
        fuse = self.fuse_siam(fuse)
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse+inp2_mul+inp1_mul+last_feature+inp1+inp2)
        out = self.fuse_siam(out)

        return out

class diff_moudel(nn.Module):
    def __init__(self,in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()
    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # weight = self.conv_1(edge)
        out = weight * x + x
        out = self.simam(out)
        return out

class CBM(nn.Module):
    def __init__(self,in_channel):
        super(CBM, self).__init__()
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        self.simam = simam_module()
    def forward(self,x1,x2):
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1-d2)
        d = self.simam(d)
        return d


class DFEM(nn.Module):
    def __init__(self,inc,outc):
        super(DFEM, self).__init__()

        self.Conv_1 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1),
                                      nn.BatchNorm2d(outc),
                                      nn.ReLU(inplace=True)
                                      )

        self.Conv = nn.Sequential(nn.Conv2d(outc, outc, kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(outc),
                                    nn.ReLU(inplace=True),
                                    )
        self.relu = nn.ReLU(inplace=True)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,diff,accom):
        cat = torch.cat([accom, diff], dim=1)
        cat = self.Conv_1(cat) + diff + accom
        c = self.Conv(cat) + cat
        c = self.relu(c) + diff
        c = self.Up(c)
        return c


class B2CNet(nn.Module):
    def __init__(self):
        super(B2CNet, self).__init__()
        self.reset = resnet18(pretrained=True)

        self.bfam_1 = BFAM(128, 32)
        self.bfam_2 = BFAM(256, 64)
        self.bfam_3 = BFAM(512, 128)
        self.bfam_4 = BFAM(1024, 256)


        self.cbm_1 = CBM(64)
        self.cbm_2 = CBM(128)
        self.cbm_3 = CBM(256)
        self.cbm_4 = CBM(512)

        self.dfem_1 = DFEM(64,32)
        self.dfem_2 = DFEM(128, 64)
        self.dfem_3 = DFEM(256, 128)
        self.dfem_4 = DFEM(512, 256)



        self.D_Conv_4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True)
                                      )
        self.D_Conv_3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True)
                                      )
        self.D_Conv_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                      )
        self.D_Conv_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      )




        self.Conv_4_c = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True)
                                      )
        self.Conv_3_c = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                      )
        self.Conv_2_c = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True)
                                      )


        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dps_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )


        self.out = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,2,kernel_size=3,padding=1))

        initialize_weights(self.bfam_1,self.bfam_2,self.bfam_3,self.bfam_4,
                           self.cbm_1,self.cbm_2,self.cbm_3,self.cbm_4,self.out,
                           self.Conv_2_c, self.Conv_3_c, self.Conv_4_c,
                           self.D_Conv_2,self.D_Conv_3,self.D_Conv_4,self.D_Conv_1,
                           self.dps_2,
                           self.dfem_1,self.dfem_2,self.dfem_3,self.dfem_4
                           )




    def forward(self,inp1,inp2):
        b1 = self.reset(inp1)
        b2 = self.reset(inp2)


        d4 = self.cbm_4(b1[3],b2[3])
        d4_2=self.D_Conv_4(d4)
        d4_1 = self.Up(d4_2)
        j4 = self.bfam_4(b1[3], b2[3])
        j4_1 = self.Up(self.Conv_4_c(j4))
        c4 = self.dfem_4(j4,d4_2)

        d3 = self.cbm_3(b1[2],b2[2])+d4_1
        d3_2 = self.D_Conv_3(d3)
        d3_1 = self.Up(d3_2)
        j3 = self.bfam_3(b1[2], b2[2], c4)+j4_1
        j3_1 = self.Up(self.Conv_3_c(j3))
        c3 = self.dfem_3(j3,d3_2)

        d2 = self.cbm_2(b1[1],b2[1])+d3_1
        d2_2 = self.D_Conv_2(d2)
        d2_1 = self.Up(d2_2)
        j2 = self.bfam_2(b1[1], b2[1], c3)+j3_1
        j2_1 = self.Up(self.Conv_2_c(j2))
        c2 = self.dfem_2(j2,d2_2)

        d1 = self.cbm_1(b1[0],b2[0])+d2_1
        d1_2 = self.D_Conv_1(d1)
        d1_1 = self.Up(d1)
        j1 = self.bfam_1(b1[0], b2[0], c2)+j2_1
        c1 = self.dfem_1(j1,d1_2)

        out2 = self.dps_2(d1_1)
        out = self.out(c1)



        return out,out2

class B2CNet_S(nn.Module):
    def __init__(self):
        super(B2CNet_S, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.layer4 = nn.Identity()

        self.bfam_1 = BFAM(128, 32)
        self.bfam_2 = BFAM(256, 64)
        self.bfam_3 = BFAM(512, 128)

        self.cbm_1 = CBM(64)
        self.cbm_2 = CBM(128)
        self.cbm_3 = CBM(256)

        self.dfem_1 = DFEM(64,32)
        self.dfem_2 = DFEM(128, 64)
        self.dfem_3 = DFEM(256, 128)



        self.D_Conv_3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True)
                                      )
        self.D_Conv_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                      )
        self.D_Conv_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      )


        self.Conv_3_c = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                      )
        self.Conv_2_c = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True)
                                      )


        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dps_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )


        self.out = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,2,kernel_size=3,padding=1))

        initialize_weights(self.bfam_1,self.bfam_2,self.bfam_3,
                           self.cbm_1,self.cbm_2,self.cbm_3,self.out,
                           self.Conv_2_c, self.Conv_3_c,
                           self.D_Conv_2,self.D_Conv_3,self.D_Conv_1,
                           self.dps_2,
                           self.dfem_1,self.dfem_2,self.dfem_3,
                           )




    def forward(self,inp1,inp2):
        c0 = self.resnet.conv1(inp1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        b1 = [c1, c2, c3]

        c0_img2 = self.resnet.conv1(inp2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)
        b2 = [c1_img2, c2_img2, c3_img2]

        d3 = self.cbm_3(b1[2],b2[2])
        d3_2 = self.D_Conv_3(d3)
        d3_1 = self.Up(d3_2)
        j3 = self.bfam_3(b1[2], b2[2])
        j3_1 = self.Up(self.Conv_3_c(j3))
        c3 = self.dfem_3(j3,d3_2)

        d2 = self.cbm_2(b1[1],b2[1])+d3_1
        d2_2 = self.D_Conv_2(d2)
        d2_1 = self.Up(d2_2)
        j2 = self.bfam_2(b1[1], b2[1], c3)+j3_1
        j2_1 = self.Up(self.Conv_2_c(j2))
        c2 = self.dfem_2(j2,d2_2)

        d1 = self.cbm_1(b1[0],b2[0])+d2_1
        d1_2 = self.D_Conv_1(d1)
        d1_1 = self.Up(d1)
        j1 = self.bfam_1(b1[0], b2[0], c2)+j2_1
        c1 = self.dfem_1(j1,d1_2)

        out2 = self.dps_2(d1_1)
        out = self.out(c1)



        return out,out2



if __name__ == '__main__':
    from thop import profile, clever_format
    inp1= torch.rand(1,3,256,256)
    inp2= torch.rand(1,3,256,256)
    model = B2CNet()
    out,ds = model(inp1,inp2)
    print(out.shape)
    print(ds.shape)


    flops, param = profile(model, [inp1, inp2])
    flops, param = clever_format([flops, param], "%.2f")
    print(flops, param)











