import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools # For partial function application in Norm layer

# --- Configuration --- (Should match data prep)
NUM_CLASSES = 23 # Including Walls(0) and Door(22)
Z_DIM = 100      # Latent noise dimension
C_DIM = 25       # Condition vector dimension (3 plot size + 22 room counts)
NGF = 64         # Base number of generator filters
NDF = 64         # Base number of discriminator filters
COND_EMBED_DIM = 64 # Dimension to embed condition vector 'c' for D
MASK_EMBED_DIM = 64 # Dimension to embed mask class IDs for D
TARGET_GEN_H = 256 # Fixed height the Generator aims to output
TARGET_GEN_W = 256 # Fixed width the Generator aims to output

# --- Helper Functions / Modules ---

def get_norm_layer(norm_type='batch'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable parameters and do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity() # Identity layer
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# --- Generator (U-Net Decoder Style) ---

class GeneratorBlock(nn.Module):
    """Defines a basic Generator block: ConvTranspose2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=nn.BatchNorm2d, use_bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class UnetGenerator(nn.Module):
    """
    Defines the U-Net based Generator.
    Takes noise z and condition c, processes them, and uses a series of
    ConvTranspose2d blocks to generate a segmentation mask logit map.
    Outputs a fixed size (TARGET_GEN_H, TARGET_GEN_W).
    """
    def __init__(self, z_dim, c_dim, num_classes, ngf=64, target_h=256, target_w=256, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_classes = num_classes
        self.ngf = ngf
        self.target_h = target_h
        self.target_w = target_w

        # Calculate the required bottleneck size based on target output size
        # target_size = bottleneck_size * 2^num_upsamples
        # Assuming bottleneck is 4x4
        self.bottleneck_h = 4
        self.bottleneck_w = 4
        num_upsamples = int(np.log2(target_h / self.bottleneck_h))
        assert target_h == target_w, "Generator currently assumes square output"
        assert target_h == self.bottleneck_h * (2**num_upsamples), "Target height must be power of 2 times bottleneck height"

        use_bias = (norm_layer == nn.InstanceNorm2d)

        # 1. Input processing layer (z + c -> bottleneck features)
        self.initial_proj = nn.Linear(z_dim + c_dim, ngf * 8 * self.bottleneck_h * self.bottleneck_w)
        self.initial_relu = nn.ReLU(True)

        # 2. Decoder / Upsampling path
        layers = []
        # Start from bottleneck channels (ngf * 8)
        in_ch = ngf * 8
        out_ch = ngf * 8
        # Add upsampling blocks. Number depends on num_upsamples.
        # Example for 256x256 output from 4x4 bottleneck (num_upsamples = 6):
        # 4x4 -> 8x8 (ngf*8 -> ngf*8)
        # 8x8 -> 16x16 (ngf*8 -> ngf*8)
        # 16x16 -> 32x32 (ngf*8 -> ngf*4)
        # 32x32 -> 64x64 (ngf*4 -> ngf*2)
        # 64x64 -> 128x128 (ngf*2 -> ngf)
        # 128x128 -> 256x256 (ngf -> ngf/2 ?? No, let's keep ngf) -> Adjust channels later

        ch_multipliers = [8] * 3 + [4, 2, 1] # Channel multipliers for ngf for the 6 steps
        for i in range(num_upsamples):
            out_ch = ngf * ch_multipliers[min(i+1, len(ch_multipliers)-1)] if i < num_upsamples-1 else ngf # Reduce channels progressively
            # For the last few layers, don't reduce below ngf
            if i > num_upsamples - 3:
                out_ch = ngf

            if i==0: in_ch = ngf*8 # First block input channels
            else: in_ch = ngf * ch_multipliers[min(i, len(ch_multipliers)-1)]
            # Special handling for last few layers to stay at ngf
            if i > num_upsamples - 4:
                 in_ch = ngf

            layers.append(GeneratorBlock(in_ch, out_ch, norm_layer=norm_layer, use_bias=use_bias))


        # 3. Final layer to map to num_classes
        # Input channels to final layer is 'out_ch' from the last GeneratorBlock (which is ngf)
        layers.append(nn.ConvTranspose2d(ngf, num_classes, kernel_size=4, stride=2, padding=1))
        # No batchnorm or ReLU here, output raw logits. Tanh is used in original Pix2Pix image output, not for logits.
        # layers.append(nn.Tanh()) # Use Tanh if outputting normalized images (-1 to 1)

        self.decoder = nn.Sequential(*layers)

    def forward(self, z, c):
        # z: [B, z_dim]
        # c: [B, c_dim]
        zc = torch.cat([z, c], dim=1)       # [B, z_dim + c_dim]
        bottleneck = self.initial_proj(zc)  # [B, ngf*8 * bottleneck_h * bottleneck_w]
        bottleneck = self.initial_relu(bottleneck)
        # Reshape to spatial tensor for the decoder
        x = bottleneck.view(-1, self.ngf * 8, self.bottleneck_h, self.bottleneck_w) # [B, ngf*8, H_bottleneck, W_bottleneck]
        # Pass through decoder blocks
        output_logits = self.decoder(x)    # [B, num_classes, target_h, target_w]
        return output_logits

# --- Discriminator (PatchGAN) ---

class DiscriminatorBlock(nn.Module):
    """Defines a basic Discriminator block: Conv2d -> BatchNorm2d -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=nn.BatchNorm2d, use_bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.block(x)

class PatchGANDiscriminator(nn.Module):
    """
    Defines the PatchGAN discriminator architecture.
    Takes a segmentation mask (class IDs) and condition vector c.
    Embeds the mask IDs, processes and replicates c, concatenates them,
    and outputs a map of patch predictions.
    """
    def __init__(self, num_classes, c_dim, ndf=64, n_layers=3, mask_embed_dim=64, cond_embed_dim=64, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            num_classes (int)   -- Number of classes in the segmentation mask (for embedding)
            c_dim (int)         -- Dimension of the condition vector c
            ndf (int)           -- Base number of discriminator filters
            n_layers (int)      -- Number of conv layers in the discriminator
            mask_embed_dim(int) -- Dimension for embedding mask class IDs
            cond_embed_dim(int) -- Dimension for embedding condition vector c spatially
            norm_layer          -- Normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.c_dim = c_dim
        self.ndf = ndf
        self.mask_embed_dim = mask_embed_dim
        self.cond_embed_dim = cond_embed_dim

        use_bias = (norm_layer == nn.InstanceNorm2d)

        # 1. Mask Embedding
        self.mask_embed = nn.Embedding(num_classes, mask_embed_dim)

        # 2. Condition Processing
        self.cond_proj = nn.Linear(c_dim, cond_embed_dim) # Project c to embedding dim

        # 3. PatchGAN Convolutional Path
        # Input channels = mask_embed_dim + cond_embed_dim
        input_nc = mask_embed_dim + cond_embed_dim
        kw = 4 # Kernel width
        padw = 1 # Padding

        # First layer (no BatchNorm)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # Middle layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # Add layers gradually increasing filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # Cap multiplier at 8 (ndf * 8 filters max)
            sequence += [
                DiscriminatorBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer, use_bias=use_bias)
            ]

        # Final layer (stride 1)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
             DiscriminatorBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, norm_layer=norm_layer, use_bias=use_bias)
        ]

        # Output layer (1 channel prediction map)
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, mask, c):
        """
        Standard forward pass.
        Parameters:
            mask (tensor) -- Input segmentation mask (Class IDs) [B, 1, H, W]
            c (tensor)    -- Condition vector [B, c_dim]
        """
        B, _, H, W = mask.shape

        # 1. Embed Mask IDs
        # Squeeze channel dim, embed, permute: [B, 1, H, W] -> [B, H, W] -> [B, H, W, mask_E] -> [B, mask_E, H, W]
        mask_embedded = self.mask_embed(mask.squeeze(1).long()) # Output: [B, H, W, mask_embed_dim]
        mask_embedded = mask_embedded.permute(0, 3, 1, 2)      # Output: [B, mask_embed_dim, H, W]

        # 2. Process and Replicate Condition Vector
        # Project c: [B, c_dim] -> [B, cond_embed_dim]
        c_processed = self.cond_proj(c)
        # Expand and replicate spatially: [B, cond_E] -> [B, cond_E, 1, 1] -> [B, cond_E, H, W]
        c_replicated = c_processed.unsqueeze(-1).unsqueeze(-1).expand(B, self.cond_embed_dim, H, W)

        # 3. Concatenate mask and condition embeddings
        combined_input = torch.cat([mask_embedded, c_replicated], dim=1) # [B, mask_E + cond_E, H, W]

        # 4. Pass through PatchGAN conv layers
        return self.model(combined_input) # [B, 1, H_patch, W_patch]


# --- Example Usage & Shape Check ---
if __name__ == '__main__':
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example parameters
    batch_size = 2
    num_classes = NUM_CLASSES
    z_dim = Z_DIM
    c_dim = C_DIM
    target_h = 128 # Example target H for testing (actual H varies)
    target_w = 128 # Example target W for testing

    # --- Test Generator ---
    print("\n--- Testing Generator ---")
    generator = UnetGenerator(z_dim, c_dim, num_classes, ngf=NGF, target_h=TARGET_GEN_H, target_w=TARGET_GEN_W).to(device)
    generator.apply(weights_init)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())/1e6:.2f} M")

    # Dummy input
    dummy_z = torch.randn(batch_size, z_dim, device=device)
    dummy_c = torch.randn(batch_size, c_dim, device=device)

    # Forward pass
    generated_logits = generator(dummy_z, dummy_c)
    print(f"Generator raw output shape: {generated_logits.shape}") # Should be [B, num_classes, TARGET_GEN_H, TARGET_GEN_W]

    # Example resizing to target H, W
    resized_logits = F.interpolate(generated_logits, size=(target_h, target_w), mode='bilinear', align_corners=False)
    print(f"Generator output resized shape: {resized_logits.shape}") # Should be [B, num_classes, target_h, target_w]

    # Get predicted class mask from resized logits
    generated_mask = torch.argmax(resized_logits, dim=1, keepdim=True) # [B, 1, target_h, target_w]
    print(f"Generated mask shape (argmax): {generated_mask.shape}")

    # --- Test Discriminator ---
    print("\n--- Testing Discriminator ---")
    discriminator = PatchGANDiscriminator(num_classes, c_dim, ndf=NDF, mask_embed_dim=MASK_EMBED_DIM, cond_embed_dim=COND_EMBED_DIM).to(device)
    discriminator.apply(weights_init)
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f} M")

    # Dummy input (use the generated mask from G)
    dummy_mask_input = generated_mask # [B, 1, target_h, target_w]

    # Forward pass
    pred_map = discriminator(dummy_mask_input, dummy_c)
    print(f"Discriminator output shape: {pred_map.shape}") # Should be [B, 1, H_patch, W_patch]