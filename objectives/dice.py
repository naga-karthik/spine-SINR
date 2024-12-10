import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    taken from: https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
    '''
    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, preds, labels):
        '''
        inputs:
            preds: logits - tensor of shape (N, H, W, ...)
            labels: soft labels [0,1] - tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        # preds = F.relu(logits) / F.relu(logits).max() if bool(F.relu(logits).max()) else F.relu(logits)
        
        numer = (preds * labels).sum()
        denor = (preds.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        # loss = - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss.mean()
    

class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # Ensure preds and labels are of the same shape
        assert preds.shape == labels.shape, "Predictions and labels must have the same shape"

        # Get unique class values
        unique_classes = torch.unique(torch.cat((torch.unique(preds), torch.unique(labels))))

        # Compute Dice loss for each class
        losses = []
        for cls in unique_classes:
            # Use soft masks instead of hard thresholding
            binary_preds = torch.where(preds == cls, preds, torch.zeros_like(preds))
            binary_labels = torch.where(labels == cls, labels, torch.zeros_like(labels))

            # ensure preds lie in [0, 1], clip to avoid negative values
            binary_preds = torch.clamp(binary_preds, 0, 1)
            binary_labels = torch.clamp(binary_labels, 0, 1)
            
            # Compute Dice loss for this class
            intersection = (binary_preds * binary_labels).sum()
            union = binary_preds.sum() + binary_labels.sum()
            
            dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            # clip the loss to avoid negative values
            dice_loss = torch.clamp(dice_loss, 0, 1)
            losses.append(dice_loss)

        # Return mean loss across classes
        return torch.stack(losses).mean()



if __name__ == "__main__":
    # Test differentiability of the loss. Key Points to verify:
    # The loss.backward() call should execute without errors.
    # The preds.grad tensor should not be None and should contain meaningful values.
    # If both conditions are met, the function is differentiable, and it works as expected with PyTorch's Autograd system.
    
    # Initialize the loss function
    loss_fn = MultiLabelDiceLoss()

    # Generate random predictions and labels
    preds = torch.rand(2, 3, 64, 64, requires_grad=True)  # Example prediction tensor
    labels = torch.randint(0, 3, (2, 3, 64, 64)).float()  # Example label tensor
    try:
        # Compute the loss
        loss = loss_fn(preds, labels)

        # Backpropagate
        loss.backward()

        # Check gradients
        print("Gradients:", preds.grad)
    except RuntimeError as e:
        print("Error during backpropagation:", e)
