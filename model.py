import torch
from torch import nn
from torch.nn import functional as F
from numpy import log2
from pyt.core_modules import Swish

class DiaConvEncoder(nn.Module):
    """get features using dilated convolutions"""
    def __init__(self, 
                 in_channels,  # num input channels
                 out_channels,  # num top-level output channels
                 hid_channels,   # num channels per hidden layer (all same for now)
                 im_size=512,   # width and height of image (must be square; must be power of 2)
                 act_fn=Swish(),
                 residual=False,
                 non_negative=False):  # only non-negative features? (encourages inference-by-parts)
        super().__init__()
        assert im_size % 2 == 0
        self.act_fn = act_fn
        self.num_hid = int(log2(im_size))
        self.residual = residual
        self.im_size = im_size
        self.non_negative = non_negative

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.readin = nn.Conv2d(in_channels, hid_channels, 1)
        convs = list()
        norms = list()
        for hi in range(self.num_hid):
            convs.append(nn.Conv2d(hid_channels, hid_channels, 3,
                                   dilation=2**hi, padding=2**hi))
            norms.append(nn.BatchNorm2d(hid_channels))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.readout = nn.Conv2d(hid_channels, out_channels, 1)

        self.w_sum_or_max_logits = nn.Parameter(torch.randn(self.out_channels))

    def forward(self, inputs):
        output = self.readin(inputs)
        for conv, norm in zip(self.convs, self.norms):
            _output = norm(conv(output))
            output = _output + output if self.residual else _output
        feature_maps = self.readout(output)
        if self.non_negative:
            feature_maps = F.softplus(feature_maps)
        batch_size = feature_maps.shape[0]
        output_flat = feature_maps.view(batch_size, self.out_channels, -1)
        sums = output_flat.sum(2)
        maxs, _ = output_flat.max(2)
        sm_weight = F.sigmoid(self.w_sum_or_max_logits)

        features = (sm_weight * sums) + ((1. - sm_weight) * maxs)
        return features, feature_maps

class DiaConvClf(nn.Module):
    """get feature maps, make reduce to sums or maxes of them, feed to classifier"""
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, inputs, return_features=False, return_feature_maps=False):
        # get feature maps
        features, feature_maps = self.encoder(inputs)
        # feed into classifier
        yhl = self.classifier(features)

        # option to return multiple things
        if not (return_features or return_feature_maps):
            output = yhl
        else:
            output = [yhl]
            if return_features:
                output.append(feature)
            if return_feature_maps:
                output.append(feature_maps)

        return output


if __name__ == '__main__':
    """example using the model on mnist images.  change params around to fit
    the size and complexity of the skin dataset"""
    from torchvision.transforms import Compose, Resize, ToTensor
    from pyt.core_modules import MLP, Identity, Swish
    from pyt.testing import test_over_mnist
    import os

    DATA_DIR = os.path.join(os.getenv('HOME'), 'datasets', 'mnist')

    # NOTE(jb to brandon): make channel and im sizes appropriate for skin
    # incread out_channels and hid_channels size for more network complexity
    IN_CHANNELS = 1
    OUT_CHANNELS = 128
    HID_CHANNELS = 64
    NUM_CLASSES = 10
    RESIDUAL = True
    IM_SIZE = 32

    # build model
    encoder = DiaConvEncoder(IN_CHANNELS, OUT_CHANNELS, HID_CHANNELS, IM_SIZE, RESIDUAL)
    classifier = MLP(OUT_CHANNELS, NUM_CLASSES, [128], act_fn=Swish())
    model = DiaConvClf(encoder, classifier)

    # NOTE: not tuned for the problem.
    # need much larger batch size for good training.  small to fit on my laptop
    training_kwargs = {'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 4}

    x_transform = Compose([
                        Resize(IM_SIZE), 
                        ToTensor()
                        ])

    test_over_mnist(model=model, 
                    data_dir=DATA_DIR, 
                    task='classify', 
                    training_kwargs=training_kwargs,
                    transform=x_transform)
