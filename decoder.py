import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from AtroustransFormer import BP_Transformer
import matplotlib.pyplot as plt 

class MAED(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(MAED, self).__init__()
        self.down_lang_layer64 = torch.nn.Sequential(
            nn.Linear(768, 64),
            nn.LayerNorm(64),
            nn.ReLU())
        # ---- ResNet Backbone ----
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x1_dem_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x6_x5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.level4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))


    def forward(self, x, x1, x2, x3, x4, x5, l4):
        # l4 is of size [24, 20, 768]
        input = x
        x1 = self.x1_dem_1(x1)
        x5_dem_1 = self.x5_dem_1(x5) # [24, 64, 20, 20]
        x4_dem_1 = self.x4_dem_1(x4) # [24, 64, 20, 20]
        x3_dem_1 = self.x3_dem_1(x3) # [24, 64, 40, 40]
        x2_dem_1 = self.x2_dem_1(x2) # [24, 64, 80, 80]

        x5_4 = self.x5_x4((1-F.sigmoid(F.upsample(x5_dem_1,size=x4.size()[2:], mode='bilinear')))*x4_dem_1)
        x4_3 = self.x4_x3((1-F.sigmoid(F.upsample(x4_dem_1,size=x3.size()[2:], mode='bilinear')))*x3_dem_1)
        x3_2 = self.x3_x2((1-F.sigmoid(F.upsample(x3_dem_1,size=x2.size()[2:], mode='bilinear')))*x2_dem_1)
        x2_1 = self.x2_x1((1-F.sigmoid(F.upsample(x2_dem_1,size=x1.size()[2:], mode='bilinear')))*x1)


        x5_4_3 = self.x5_x4_x3((1-F.sigmoid(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')))*x4_3)
        x4_3_2 = self.x4_x3_x2((1-F.sigmoid(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')))*x3_2)
        x3_2_1 = self.x3_x2_x1((1-F.sigmoid(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')))*x2_1)


        x5_4_3_2 = self.x5_x4_x3_x2((1-F.sigmoid(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')))*x4_3_2)
        x4_3_2_1 = self.x4_x3_x2_x1((1-F.sigmoid(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')))*x3_2_1)

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1((1-F.sigmoid(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')))*x4_3_2_1)

        level4 = self.level3(x5_4 + x4_dem_1)
        level3 = self.level3(x4_3 + x5_4_3+ x3_dem_1)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2+ x2_dem_1)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1+x1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)

        output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        return output

