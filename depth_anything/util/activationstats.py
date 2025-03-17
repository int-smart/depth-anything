from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import numpy as np

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
            act_flat = activation.view(activation.size(0), -1)
            
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
    
    def close(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
