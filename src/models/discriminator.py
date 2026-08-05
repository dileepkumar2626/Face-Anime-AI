import torch
import torch.nn as nn
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, n_filters=64):
        super().__init__()
 
        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
 
        model = []
        model += conv_block(input_nc, n_filters, stride=2, norm=False)       
        model += conv_block(n_filters, n_filters * 2, stride=2)            
        model += conv_block(n_filters * 2, n_filters * 4, stride=2)         
        model += conv_block(n_filters * 4, n_filters * 8, stride=1)
        model += [nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=1, padding=1)]
 
        self.model = nn.Sequential(*model)
 
    def forward(self, x):
        return self.model(x)
if __name__ == "__main__":
    D = PatchDiscriminator()
    dummy = torch.randn(2, 3, 256, 256)
    out = D(dummy)
    print("Discriminator output shape:", out.shape)