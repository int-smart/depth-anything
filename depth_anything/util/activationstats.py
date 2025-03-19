from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class ActivationStats:
    def __init__(self, writer, model, layer_names=None):
        """
        Track activation statistics for specific layers in a model
        
        Args:
            writer: TensorBoard SummaryWriter
            model: PyTorch model
            layer_names: List of layer names to track (if None, track all Conv2d and ConvTranspose2d)
        """
        self.writer = writer
        self.model = model
        self.hooks = []
        self.activations = {}
        self.layer_names = layer_names
        
        # Register hooks for all convolutional layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on conv layers"""
        for name, module in self.model.named_modules():
            # Only track Conv2d and ConvTranspose2d layers
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # If layer_names is specified, only track those layers
                if self.layer_names is None or name in self.layer_names:
                    self.hooks.append(
                        module.register_forward_hook(
                            lambda m, i, o, name=name: self._hook_fn(name, o)
                        )
                    )
    
    def _hook_fn(self, name, output):
        """Hook function to store activations"""
        self.activations[name] = output.detach()
    
    def log_stats(self, step):
        """Log activation statistics to TensorBoard"""
        for name, activation in self.activations.items():
            # Calculate statistics
            act_flat = activation.reshape(activation.size(0), -1)
            
            # Log scalar statistics
            self.writer.add_scalar(f'act_stats/{name}/mean', act_flat.mean().item(), step)
            self.writer.add_scalar(f'act_stats/{name}/std', act_flat.std().item(), step)
            self.writer.add_scalar(f'act_stats/{name}/min', act_flat.min().item(), step)
            self.writer.add_scalar(f'act_stats/{name}/max', act_flat.max().item(), step)
            
            # Log histogram of activations
            self.writer.add_histogram(f'act_hist/{name}', act_flat, step)
            
            # Log sparsity (percentage of zeros)
            zeros = (act_flat == 0).float().mean().item() * 100
            self.writer.add_scalar(f'act_stats/{name}/sparsity_pct', zeros, step)

    def visualize_activations(self, step, max_channels=64, batch_idx=0):
        """
        Visualize activations by creating a grid of feature maps and log to TensorBoard
        
        Args:
            step: Global step for TensorBoard logging
            max_channels: Maximum number of channels to visualize per layer
            batch_idx: Which batch item to visualize
        """
        for name, activation in self.activations.items():
            # Skip activations that don't have the right shape
            if len(activation.shape) != 4:
                continue
            
            # Create activation grid visualization
            img_array = self._create_activation_grid(activation, max_channels, batch_idx)
            
            # Log to TensorBoard
            self.writer.add_image(f'act_vis/{name}', img_array, step, dataformats='HWC')
    
    def _create_activation_grid(self, activations, max_channels=64, batch_idx=0):
        """
        Create a grid visualization of activation channels
        
        Args:
            activations: Tensor of shape [B, C, H, W]
            max_channels: Maximum number of channels to visualize
            batch_idx: Which batch item to visualize
        
        Returns:
            Numpy array of the composite image in HWC format
        """
        # Select the specified batch item
        act = activations[batch_idx]
        
        # Get dimensions
        C, H, W = act.shape
        num_channels = min(C, max_channels)
        
        # Determine grid size based on number of channels
        if num_channels <= 16:
            grid_size = (4, 4)
        elif num_channels <= 64:
            grid_size = (8, 8)
        elif num_channels <= 192:
            grid_size = (12, 16)
        else:
            grid_size = (24, 16)
        
        grid_rows, grid_cols = grid_size
        
        # Create a grid to hold all channels
        grid_h = grid_rows * H
        grid_w = grid_cols * W
        grid = np.zeros((grid_h, grid_w))
        
        # Fill the grid with activation channels
        for idx in range(min(num_channels, grid_rows * grid_cols)):
            # Calculate grid position
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Extract the channel
            channel = act[idx].cpu().numpy()
            
            # Normalize the channel for better visualization
            if channel.max() != channel.min():
                channel = (channel - channel.min()) / (channel.max() - channel.min())
            
            # Place in grid
            grid[row*H:(row+1)*H, col*W:(col+1)*W] = channel
        
        # Convert to RGB using colormap
        plt.figure(figsize=(20, 20))
        plt.imshow(grid, cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        
        # Convert figure to numpy array
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        return img_array
    def close(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
