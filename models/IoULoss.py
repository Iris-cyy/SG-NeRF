import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, sample_dist, resolution=256, topk=16):
        super(IoULoss, self).__init__()
        self.sample_dist = sample_dist
        self.resolution = resolution
        self.equals = complex(0, resolution)
        self.topk = topk
        self.sigma = 6 / self.resolution
        equals = torch.linspace(-1, 1, self.resolution)
        x, y, z = torch.meshgrid(equals, equals, equals)
        samples = torch.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
        self.sample_2 = samples.expand(1, topk, -1, -1)
        self.pdist = nn.PairwiseDistance(p=2)
        self.sigmoid_param = 30

    def Gaussian(self, means, weights): #batchsize, topk, 1, 3 / batchsize, topk, 1
        diff = self.sample_2 - means #batchsize, topk, res^3, 3
        squared_distances = (diff ** 2).sum(dim=-1)
        gauss = torch.exp(-squared_distances / (2 * self.sigma * self.sigma)) #batchsize, topk, res^3
        gauss = (gauss * weights).sum(dim=1) # batchsize, res^3
        return gauss

    def mixGaussian(self, pts, weights):
        heatmap = self.Gaussian(pts[:, :, None, :], weights[:, :, None])
        return heatmap

    def forward(self, pts1, weights1, pts2, weights2):
        idx = ~torch.isnan(weights1).any(dim=1) & ~torch.isnan(weights2).any(dim=1)
        pts1 = pts1[idx]
        pts2 = pts2[idx]
        weights1 = weights1[idx]
        weights2 = weights2[idx]

        if len(pts1) == 0:
            return -1

        if len(weights1.shape) == 1:
            pts1.unsqueeze(dim=0)
            weights1.unsqueeze(dim=0)
        if len(weights2.shape) == 1:
            pts2.unsqueeze(dim=0)
            weights2.unsqueeze(dim=0)

        self.n_batches = pts1.shape[0]
        self.n_sample = pts1.shape[1]

        top_v1, top_i1 = torch.topk(weights1, k=self.topk, dim=1, largest=True)
        expanded_i1 = top_i1.unsqueeze(dim=2)
        weights1 = top_v1
        weights1 /= (weights1.sum(dim=1)[:, None] + 1e-5)
        pts1 = torch.gather(pts1, 1, expanded_i1.expand(-1, -1, 3))

        top_v2, top_i2 = torch.topk(weights2, k=self.topk, dim=1, largest=True)
        expanded_i2 = top_i2.unsqueeze(dim=2)
        weights2 = top_v2
        weights2 /= (weights2.sum(dim=1)[:, None] + 1e-5)
        pts2 = torch.gather(pts2, 1, expanded_i2.expand(-1, -1, 3))

        x_min = float(min(pts1[:, :, 0].min(), pts2[:, :, 0].min())) - 3*self.sigma
        y_min = float(min(pts1[:, :, 1].min(), pts2[:, :, 1].min())) - 3*self.sigma
        z_min = float(min(pts1[:, :, 2].min(), pts2[:, :, 2].min())) - 3*self.sigma
        x_max = float(max(pts1[:, :, 0].max(), pts2[:, :, 0].max())) + 3*self.sigma
        y_max = float(max(pts1[:, :, 1].max(), pts2[:, :, 1].max())) + 3*self.sigma
        z_max = float(max(pts1[:, :, 2].max(), pts2[:, :, 2].max())) + 3*self.sigma
        equal_x = torch.linspace(x_min, x_max, self.resolution)
        equal_y = torch.linspace(y_min, y_max, self.resolution)
        equal_z = torch.linspace(z_min, z_max, self.resolution)
        x, y, z = torch.meshgrid(equal_x, equal_y, equal_z)
        samples = torch.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
        self.sample_2 = samples.expand(1, self.topk, -1, -1)

        heatmap1 = self.mixGaussian(pts1, weights1)
        if heatmap1 is None:
            return -1
        heatmap2 = self.mixGaussian(pts2, weights2)
        if heatmap2 is None:
            return -1

        I = (heatmap1 * heatmap2).sum()
        U = (heatmap1 + heatmap2 - heatmap1 * heatmap2).sum() + 1e-5
        iou = I / U

        return 1 - iou
