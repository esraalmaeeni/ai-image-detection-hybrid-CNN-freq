import torch
import torchvision
from torch import nn

def set_seeds(seed: int=42):
    """Sets random sets for torch operations."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalLayerFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambda_
        return grad_input, None

class BaseFeatureExtractor(nn.Module):
    def __init__(self, input_channels, feature_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        return x

class SpatialBranch(BaseFeatureExtractor):
    def __init__(self, feature_dim=256):
        super().__init__(input_channels=3, feature_dim=feature_dim)

class FrequencyBranch(BaseFeatureExtractor):
    def __init__(self, feature_dim=256):
        super().__init__(input_channels=1, feature_dim=feature_dim)

class GatedFusionModule(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.att_weight_spatial = nn.Parameter(torch.ones(feature_dim))
        self.att_weight_freq = nn.Parameter(torch.ones(feature_dim))

        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, spatial_features, frequency_features):
        att_s = torch.sigmoid(self.att_weight_spatial)
        att_f = torch.sigmoid(self.att_weight_freq)
        spatial_weighted = spatial_features * att_s
        frequency_weighted = frequency_features * att_f

        combined_features = torch.cat((spatial_weighted, frequency_weighted), dim=1)

        x = self.dropout(F.relu(self.fc1(combined_features)))
        output = self.fc2(x)

        return output, x

class AdversarialDiscriminator(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int = 64, num_domains: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_domains)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, feature_dim=256, fusion_hidden_dim=128, disc_hidden_dim=64, lambda_grl=0.1, use_adversarial=True, num_domains=1):
        super().__init__()
        self.use_adversarial = use_adversarial
        self.lambda_grl = lambda_grl

        self.spatial_branch = SpatialBranch(feature_dim)
        self.frequency_branch = FrequencyBranch(feature_dim)
        self.gated_fusion = GatedFusionModule(feature_dim, fusion_hidden_dim)

        if self.use_adversarial:
            self.domain_discriminator = AdversarialDiscriminator(fusion_hidden_dim, disc_hidden_dim, num_domains)

    def forward(self, spatial_input, frequency_input):
        spatial_features = self.spatial_branch(spatial_input)
        frequency_features = self.frequency_branch(frequency_input)

        main_task_output, intermediate_features = self.gated_fusion(spatial_features, frequency_features)

        if self.use_adversarial:
            reversed_features = GradientReversalLayerFn.apply(intermediate_features, self.lambda_grl)
            domain_task_output = self.domain_discriminator(reversed_features)
            return main_task_output, domain_task_output

        return main_task_output

# ==========================================
# Testing and Verification
# ==========================================

def get_fft_representation(rgb_batch):
    """
    Properly converts an RGB image batch to a 1-channel log-magnitude FFT spectrum.
    """
    gray = (0.2989 * rgb_batch[:, 0:1, :, :]) + (0.5870 * rgb_batch[:, 1:2, :, :]) + (0.1140 * rgb_batch[:, 2:3, :, :])

    fft = torch.fft.fft2(gray, norm="ortho")
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    
    magnitude = torch.abs(fft_shift)
    log_magnitude = torch.log(magnitude + 1e-8)

    return log_magnitude

if __name__ == "__main__":
    feature_dim = 256
    fusion_hidden_dim = 128

    spatial_input = torch.randn(4, 3, 224, 224)
    frequency_input = get_fft_representation(spatial_input)

    print("--- Testing Baseline Model (No Discriminator) ---")
    baseline_model = HybridModel(use_adversarial=False)
    main_output = baseline_model(spatial_input, frequency_input)
    print("Main Task Output Shape:", main_output.shape)

    print("\n--- Testing Domain-Adversarial Model ---")
    adv_model = HybridModel(use_adversarial=True, lambda_grl=0.1, num_domains=4)
    main_output, domain_output = adv_model(spatial_input, frequency_input)

    print("Main Task Output Shape:", main_output.shape) 
    print("Domain Task Output Shape:", domain_output.shape)
