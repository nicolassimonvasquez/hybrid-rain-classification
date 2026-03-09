import torch
import torch.nn as nn
from typing import Dict, Tuple
from . import create_backbone, create_opt_backbone
from . import create_head, create_opt_head

class Recognizer2D(nn.Module):
    """2D recognizer model framework.
    Args:
        backbone: Backbone modules to extract feature.
        cls_head: Classification head to process feature
    """
    def __init__(self,
                 backbone: str,
                 cls_head: str,
                 num_labels: int,
                 dropout_rate: float,
                 is_3d:bool
                 ) -> None:
        super(Recognizer2D, self).__init__()
        if backbone=='tsm':
            self.backbone, in_feats = create_backbone(backbone)
        else:
            raise NotImplementedError(f"{backbone} is not implemented for 2D recognizer.")
        self.cls_head = create_head(name=cls_head,
                                    num_labels=num_labels,
                                    in_feats=in_feats,
                                    dropout_rate=dropout_rate,
                                    is_3d=is_3d)

    def forward(self, inputs: torch.Tensor) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (Tensor): The input data. BTCHW

        Returns:
            Tensor: The extracted logits.
        """
        if inputs.shape[1] == 3 and inputs.shape[2] != 3:
            inputs = inputs.permute(0, 2, 1, 3, 4)
        num_segs = inputs.shape[1]
        # [N, T, C, H, W] -> # [N * T, C, H, W]
        inputs = inputs.reshape((-1, ) + inputs.shape[2:]) 

        x = self.backbone(inputs) # (N * T, C, H, W)
        
        # Apply Global Average Pooling (Gap)
        if x.dim() == 4:
            x = torch.mean(x, dim=(2, 3)) # (N * T, C)
            
        x = self.cls_head(x) # (N * T, num_classes) and tuple handling if mcmh
        
        # Handle tuple output from head (mcmh)
        if isinstance(x, tuple):
             # Tuple of tensors: (rx, fx, sx) each is [N*T, 3]
             # We need to reshape each
             outs = []
             for out in x:
                 out = out.reshape((-1, num_segs) + out.shape[1:]) # (N, T, 3)
                 out = torch.mean(out, dim=1) # (N, 3) Consensus
                 outs.append(out)
             if len(outs) == 1: return outs[0]
             return tuple(outs)
        else:
             x = x.reshape((-1, num_segs) + x.shape[1:]) # (N, T, num_classes)
             x = torch.mean(x, dim=1) # (N, num_classes) Consensus
             return x

class Recognizer3D(nn.Module):
    """3D recognizer model framework.
    Args:
        backbone: Backbone modules to extract feature.
        cls_head: Classification head to process feature
    """
    __valid_models = ['x3d', 'i3d', 'swin', 'mvit', 'r3d_18']
    def __init__(self,
                 backbone: str,
                 cls_head: str,
                 num_labels: int,
                 dropout_rate: float,
                 is_3d:bool # for the input to the class
                 ) -> None:
        super(Recognizer3D, self).__init__()
        if backbone in self.__valid_models:
            self.backbone, in_feats = create_backbone(backbone)
        else:
            raise NotImplementedError(f"{backbone} is not implemented for 2D recognizer.")
        self.cls_head = create_head(name=cls_head,
                                    num_labels=num_labels,
                                    in_feats=in_feats,
                                    dropout_rate=dropout_rate,
                                    is_3d=is_3d)

    def forward(self, inputs: torch.Tensor) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (Tensor): The input data. BCTHW

        Returns:
                Tensor: The extracted logits.
        """
        x = self.backbone(inputs) # (N, C, T, H, W)
        x = self.cls_head(x)
        return x

class MMRecognizer3D(nn.Module):
    """
    Multi-modal 3D recognizer model framework. 
    Only for TSM ResNet50.
    Class Head: Average, Conv3D, Channel Attention"""
    def __init__(self,
                 consensus: str,
                 cls_head: str,
                 num_labels: int,
                 dropout_rate: float,
                 is_3d:bool
                 ) -> None:
        super(MMRecognizer3D, self).__init__()
        self.backboneS, in_feats = create_backbone('tsm')
        self.backboneT = create_opt_backbone()
        self.cls_head = create_opt_head(consensus=consensus,
                                        name=cls_head,
                                        num_labels=num_labels,
                                        in_feats=in_feats,
                                        dropout_rate=dropout_rate,
                                        is_3d=is_3d)

    def forward(self, xS, xT) -> tuple:
        """Extract features.

        Args:
            inputs (dict[str, torch.Tensor]): The multi-modal input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'backbone'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                tuple[torch.Tensor]: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline.
        """
        num_segsS = xS.shape[1]
        xS = xS.view((-1, ) + xS.shape[2:]) # [N, T, C, H, W] -> # [N * T, C, H, W]
        xS = self.backboneS(xS) # (N * T, C, H, W)
        xS = xS.reshape((-1, num_segsS) + xS.shape[1:]).transpose(1, 2).contiguous() #(N, C, T, H, W)
        
        num_segsT = xT.shape[1]
        xT = xT.view((-1, ) + xT.shape[2:]) # [N, T, C, H, W] -> # [N * T, C, H, W]
        xT = self.backboneT(xT) # (N * T, C, H, W)
        xT = xT.reshape((-1, num_segsT) + xT.shape[1:]).transpose(1, 2).contiguous() #(N, C, T, H, W)
        
        x = self.cls_head(xS, xT)
        return x