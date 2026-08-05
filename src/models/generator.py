import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_filters=64, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, n_filters, kernel_size=7),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU(inplace=True),
        ]
        n_downsampling = 2
        curr_filters = n_filters
        for _ in range(n_downsampling):
            model += [
                nn.Conv2d(curr_filters, curr_filters * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_filters * 2),
                nn.ReLU(inplace=True),
            ]
            curr_filters *= 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(curr_filters)]
        for _ in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(
                    curr_filters, curr_filters // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.InstanceNorm2d(curr_filters // 2),
                nn.ReLU(inplace=True),
            ]
            curr_filters //= 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_filters, output_nc, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
if __name__ == "__main__":
    G = ResnetGenerator(n_residual_blocks=6)
    dummy = torch.randn(2, 3, 256, 256)
    out = G(dummy)
    print("Generator output shape:", out.shape)