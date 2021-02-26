# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:01 PM 2/25/2021 
# --------------------------------------------------------
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from backbone import VisionTransformerExtractor
import torch
from torch import nn

backbone = VisionTransformerExtractor()
img = torch.randn(2, 3, 1024, 1024)

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
print(backbone(img).shape)

backbone.out_channels =1280 # 768 #

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model

#model(img)

class ViTRFCNN(nn.Module):
    def __init__(self, name='vit_base_patch32_384', img_size= (800, 800), patch_size=32, num_classes=0, in_chans=3):
        super().__init__()
        self.backbone = VisionTransformerExtractor(name, img_size, patch_size)
        self.model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    def forward(self, x):
        return self.model(x)



