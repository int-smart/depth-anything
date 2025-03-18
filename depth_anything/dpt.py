import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from HRWSI.data.hrwsi import get_hrwsi_loader
from blocks import FeatureFusionBlock, _make_scratch
from util.activationstats import ActivationStats

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            # For layers with ReLU activation
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            # print(f"Feature {i} before reshape: {x.shape}")
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            # print(f"Feature {i} after reshape: {x.shape}")

            x = self.projects[i](x)
            # print(f"Feature {i} after projection: {x.shape}")
            x = self.resize_layers[i](x)
            # print(f"Feature {i} after resize: {x.shape}")

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        # print(f"Layer shapes: 1:{layer_1.shape}, 2:{layer_2.shape}, 3:{layer_3.shape}, 4:{layer_4.shape}")

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # print(f"RN layer shapes: 1:{layer_1_rn.shape}, 2:{layer_2_rn.shape}, 3:{layer_3_rn.shape}, 4:{layer_4_rn.shape}")

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        # print(f"Path 4 shape: {path_4.shape}")
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # print(f"Path 3 shape: {path_3.shape}")
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # print(f"Path 2 shape: {path_2.shape}")
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(f"Path 1 shape: {path_1.shape}")

        out = self.scratch.output_conv1(path_1)
        # print(f"After output_conv1: {out.shape}")
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # print(f"After interpolation: {out.shape}")
        out = self.scratch.output_conv2(out)
        # print(f"Final output shape: {out.shape}")

        return out

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):

        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()

        assert encoder in ['vits', 'vitb', 'vitl']

        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.loss = ScaleAndShiftInvariantLoss()

    def forward(self, x, y):
        # print(f"Input image shape: {x.shape}")
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        # print(f"Features after intermediate layers: {len(features)} feature maps with features {features[0][0].shape}, {features[1][0].shape}, {features[2][0].shape}, {features[3][0].shape},{features[0][1].shape}, {features[1][1].shape}, {features[2][1].shape}, {features[3][1].shape}")

        patch_h, patch_w = h // 14, w // 14
        # print(f"Patch dimensions: {patch_h}x{patch_w}")

        depth = self.depth_head(features, patch_h, patch_w)
        # print(f"Depth head output shape: {depth.shape}")

        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        # print(f"After interpolation shape: {depth.shape}")

        depth = F.relu(depth)
        # print(f"Final depth shape before squeeze: {depth.shape}")

        loss = None
        if y is not None:
            loss = self.loss(depth, y, torch.ones_like(depth, dtype=torch.bool))
        return depth.squeeze(1), loss


class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)

def fix_random_seed(seed: int):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def configure_optimizer(model, learning_rate, weight_decay, device_type, max_epochs=100, min_lr_ratio=0.1, warmup_epochs=5):
    import inspect
    import math
    from torch.optim.lr_scheduler import LambdaLR

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    slow_learn_params = [p for pn, p in param_dict.items() if "pretrained" in pn]
    fast_learn_params = [p for pn, p in param_dict.items() if "depth_head" in pn]
    # Collect any remaining parameters
    remaining_params = [p for pn, p in param_dict.items()
                       if "pretrained" not in pn and "depth_head" not in pn]

    optim_groups = [
        {"params": slow_learn_params, "lr": learning_rate*0.1, "weight_decay": 0},
        {"params": fast_learn_params, "lr": learning_rate, "weight_decay": weight_decay},
    ]
    # Add remaining parameters with default settings if any exist
    if remaining_params:
        optim_groups.append({"params": remaining_params, "lr": learning_rate, "weight_decay": weight_decay})

    num_slow_params = sum(p.numel() for p in slow_learn_params)
    num_fast_params = sum(p.numel() for p in fast_learn_params)
    if master_process:
        print(f"num slow learning parameter tensors: {len(slow_learn_params)}, with {num_slow_params:,} parameters")
        print(f"num fast learning parameter tensors: {len(fast_learn_params)}, with {num_fast_params:,} parameters")
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optim = torch.optim.AdamW(
        optim_groups, fused=use_fused
    )
    # Define a lambda function for the learning rate schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    # Create scheduler with the lambda function
    scheduler = LambdaLR(optim, lr_lambda)


    if master_process:
        print(f"Using cosine learning rate scheduler with {warmup_epochs} warmup epochs")
        print(f"Initial learning rate: {learning_rate}, Final learning rate: {learning_rate*min_lr_ratio}")

    return optim, scheduler



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="batch size for training"
    )
    parser.add_argument(
        "--data_dir",
        default="/kaggle/input/depthdata",
        type=str,
        help="directory containing the training data"
    )    
    args = parser.parse_args()
    
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        master_process = True
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    max_epochs = 2
    batch_size = args.batch_size
    dataset_size = 20378//(batch_size*ddp_world_size)
    warmup_steps = 0.33*dataset_size*max_epochs
    max_steps = dataset_size*max_epochs
    
    # Create a directory to save the images if it doesn't exist
    os.makedirs("prediction_images", exist_ok=True)

    fix_random_seed(42)
    torch.set_float32_matmul_precision("high")

    # Create the dataloader
    # Create dataloader with DDP support
    dataloader = get_hrwsi_loader(
        data_dir_root=args.data_dir,
        resize_shape=(518,518),  # Example resize shape
        batch_size=batch_size,  # Adjust as needed
        ddp=True,  # Pass the DDP flag
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )

    model = DepthAnything({'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'use_bn': False, 'use_clstoken': False})
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model
    # model = torch.compile(model)
    optimizer, scheduler = configure_optimizer(raw_model, 5e-5, 0.1, device_type, max_epochs=max_steps, min_lr_ratio=0.1, warmup_epochs=warmup_steps)
    # Create a SummaryWriter instance
    writer = SummaryWriter(log_dir="runs/depth_anything_experiment")
    act_stats = ActivationStats(writer, model, None)

    for epoch in range(max_epochs):
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > 0 and batch_idx % 10 == 0:
                model.eval()
                with torch.no_grad():
                    for val_idx, batch in enumerate(dataloader):
                        if val_idx >= 1:  # Limit to 5 validation samples to avoid too many images
                            break
                        image = batch['image']  # Shape: [B,C,H,W]
                        depth = batch['depth']

                        image = image.to(device)
                        depth = depth.to(device)

                        pred, loss = model(image, depth)
                        # Save pred as image
                        # Process each image in the batch
                        for i in range(image.shape[0]):
                            # Get the current prediction and convert to CPU numpy array
                            pred_np = pred[i].cpu().numpy()
                            
                            # Normalize to 0-1 range if not already
                            if pred_np.max() > 1.0 or pred_np.min() < 0.0:
                                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                            
                            # Convert to uint8 for image saving (0-255)
                            pred_img = (pred_np * 255).astype(np.uint8)
                            
                            # If prediction has a channel dimension, remove it for grayscale image
                            if len(pred_img.shape) > 2:
                                pred_img = pred_img.squeeze()

                            # Save using PIL
                            img = Image.fromarray(pred_img)
                            img.save(f"prediction_images/pred_batch{batch_idx}_sample{val_idx}_{i}.png")
                            
                            # Optionally, also save the ground truth for comparison
                            depth_np = depth[i].cpu().numpy().squeeze()
                            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                            depth_img = (depth_np * 255).astype(np.uint8)
                            
                            depth_pil = Image.fromarray(depth_img)
                            depth_pil.save(f"prediction_images/gt_batch{batch_idx}_sample{val_idx}_{i}.png")
                            
                            # Optionally, save the input image for reference
                            input_img = image[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                            # Denormalize if your images were normalized
                            input_img = input_img * np.array([0.2402, 0.2401, 0.2459]) + np.array([0.4268, 0.4177, 0.3832])
                            input_img = np.clip(input_img, 0, 1)
                            
                            plt.imsave(f"prediction_images/input_batch{batch_idx}_sample{val_idx}_{i}.png", input_img)

                        if master_process:
                            # Log input, prediction, gt images
                            writer.add_images('Prediction', pred.unsqueeze(1)  , epoch * len(dataloader) + batch_idx)
                            writer.add_images('Input', image, epoch * len(dataloader) + batch_idx)
                            writer.add_images('GT Depth', depth, epoch * len(dataloader) + batch_idx)
                            act_stats.log_stats(epoch * len(dataloader) + batch_idx)
                            print(f"Saved prediction images for batch {batch_idx}, validation sample {val_idx}")
                

            model.train()
            optimizer.zero_grad()
            loss_accum = 0.0

            image = batch['image']  # Shape: [B,C,H,W]
            depth = batch['depth']

            image = image.to(device)
            depth = depth.to(device)

            # Mixed precision training
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                pred, loss = model(image, depth)
            loss_accum += loss.detach()
            
            loss.backward()
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if master_process:
                # Log scalar values (loss)
                writer.add_scalar('Loss/train', loss_accum.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar('Gradient/norm', norm, epoch * len(dataloader) + batch_idx)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch * len(dataloader) + batch_idx)
            
            optimizer.step()
            scheduler.step()
            if master_process:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss_accum.item():.4f}, norm: {norm:.4f}")

    # Close the writer when done
    writer.close()
    # Clean up hooks when done
    act_stats.close()
    
    if ddp:
        destroy_process_group()
    # torch.Size([3, 642, 900])
    # Dataset mean: tensor([0.4268, 0.4177, 0.3832])
    # Dataset std: tensor([0.2402, 0.2401, 0.2459])
    