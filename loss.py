import torch
import torch.nn.functional as F


#converting this into a callable class for easy parameterization
class BCEWithFocalLoss(object):
    """Returns BCE with focal loss

    Args: 
        alpha: class imbalance param (hyper-parameter chosen to in [0.6, 0.7, 0.8, 0.9])
        gamma: focus param (hyper-parameter chosen to be in [0,1,2,3,4,5])
        min_thresh: fixed to be 0.01. Threshold to pick 'active' voxels
        reduction: how you want the loss to be reduced.    
        preds: model output (without sigmoid activation)
        gt_patch: ground-truth heatmap
    """
    
    def __init__(self, alpha, gamma, min_thresh, reduction):
        self.alpha = alpha
        self.gamma = gamma
        self.min_thresh = min_thresh 
        self.reduction = reduction
        
    def __call__(self, preds, gt_patch):
        #both BCE with logits and softplus work directly on logits as they implement the sigmoid layer themselves

        #1st part (+ve part of the BCE)
        bce_loss = 1/2*gt_patch*F.softplus(-preds)

        #2nd part (focal loss part)
        alpha_t = torch.where(gt_patch > self.min_thresh, self.alpha, 1-self.alpha)         
        ce_loss = F.binary_cross_entropy_with_logits(preds, 
                                                     (gt_patch > self.min_thresh).float(), #binarizing the ground-truth heatmap
                                                     reduction="none")
        
        pred_sigmoid = torch.sigmoid(preds) 
        h_t = torch.where(gt_patch > self.min_thresh, pred_sigmoid, 1-pred_sigmoid)         
        fc_loss = 1/2*alpha_t*((1-h_t)**self.gamma)*ce_loss

        loss =  bce_loss + fc_loss 
  
        #Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
            bce_loss = bce_loss.mean()
            fc_loss = fc_loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            bce_loss = bce_loss.sum()
            fc_loss = fc_loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        return loss

