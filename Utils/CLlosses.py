import torch
import torch.nn as nn
import torch.nn.functional as F

def drlim_loss(rep, target, margin, device='cuda'):
    distances = F.pairwise_distance(rep[0::2], rep[1::2])
    loss_similar = target * distances**2  
    loss_dissimilar = (1 - target) * F.relu(margin - distances)**2  
    loss = torch.mean(loss_similar + loss_dissimilar)
    return loss


def Rasd_loss(x, y, centroids):
    unique_y = torch.unique(y)
    extended_x = x.unsqueeze(0)
    extended_centroids = centroids[unique_y.long()].unsqueeze(1)
    dists = torch.norm(extended_x - extended_centroids, dim=2)
    masks = y == unique_y[:, None]
    dists_masked = dists * masks
    dists_sq_sum = (dists_masked**2).sum(1)
    counts = masks.sum(1).clamp_min(1)
    dists_avg = dists_sq_sum / counts.float()
    loss = torch.mean(dists_avg)
    return loss

def LSL_loss(projection, target, alpha):
    D = torch.square(torch.cdist(projection, projection))
    target_long = target.long()
    classes = torch.unique(target_long)
    bin_counts = torch.bincount(target_long)
    P = torch.zeros_like(classes)
    for idx, c in enumerate(classes):
        P[idx] = bin_counts[c.item()]
    p = [torch.where(target_long == c)[0] for c in classes]
    A = [D[i][:, i] for i in p]
    n = [torch.where(target_long != c)[0] for c in classes]
    B = [torch.sum(torch.exp(alpha - D[i.view(-1, 1), j.view(1, -1)]), dim=1) for i, j in zip(p, n)]
    C = [torch.sum(torch.exp(alpha - D[i.view(-1, 1), j.view(1, -1)]), dim=1) for i, j in zip(p, n)]
    loss = sum((torch.sum(torch.square(torch.maximum(a + torch.log(b + c), torch.tensor(0.0, dtype=torch.float64)))) / (2.0 * p_count) for a, b, c, p_count in zip(A, B, C, P)))
    return loss / len(target)


def CADE_loss(model, x1, x2, y1, y2, margin):
    criterion = nn.MSELoss()
    outputs1 = model(x1)
    outputs2 = model(x2)
    recon_loss = criterion(outputs1, x1) + criterion(outputs2, x2)
    dist = torch.sqrt(torch.sum((model.module.encoder(x1) - model.module.encoder(x2))**2, dim=1) + 1e-10)
    is_same = (y1 == y2).float()
    contrastive_loss = torch.mean(is_same * dist + (1 - is_same) * torch.relu(margin - dist))
    return contrastive_loss, recon_loss
