import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=2):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.LSTM(input_size, hidden_size, cell_num, batch_first=True)

    def forward(self, feats):
        h0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        c0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # aggregated feature
        feat, _ = self.rnn(feats, (h0, c0))
        return feat[:,-1,:]

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 128)
        self.AFeatPool = FeatAggregate(128, 128)
        self.fc = nn.Linear(128, 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc(afeat)
        return F.pairwise_distance(vfeat, afeat)


# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = vfeat.transpose(2, 1)
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = F.relu(self.vfc(vfeat))
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = afeat.transpose(2, 1)
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)
        return F.pairwise_distance(vfeat, afeat)

class MyVAMetric(nn.Module):
    def __init__(self, framenum=120):
        super(MyVAMetric, self).__init__()
        #self.vconv1=nn.Conv1d(in_channels=120, out_channels=120, kernel_size=5, stride=1, padding=2)
        self.vconv=nn.Conv1d(1024,128,5,1,2)
        self.aconv=nn.Conv1d(128,128,5,1,2)
        self.ap=nn.AvgPool1d(120)
        #self.apt=nn.AvgPool1d(8)
        #self.fct=nn.Linear(15,1)
        #self.vfc=nn.Linear(1024,128)
        self.vl=nn.Linear(128,96)
        self.al=nn.Linear(128,96)
        self.val=nn.Linear(96,96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat=vfeat.transpose(2,1)
        vfeat=F.relu(self.vconv(vfeat))
        vfeat=self.ap(vfeat)
        #vfeat=F.relu(self.fct(vfeat))
        #vfeat=self.fct2(vfeat)
        #vfeat=vfeat.transpose(2,1)
        #vfeat=self.apt(vfeat)
        #vfeat=vfeat.transpose(2,1)
        vfeat=vfeat.view(-1,128)
        #vfeat=F.relu(self.vfc(vfeat))
        vfeat=F.sigmoid(self.vl(vfeat))
        vfeat=self.val(vfeat)


        afeat=afeat.transpose(2,1)
        afeat=F.relu(self.aconv(afeat))
        afeat=self.ap(afeat)
        #afeat=F.relu(self.fct(afeat))
        #afeat=self.fct2(afeat)
        afeat=afeat.view(-1,128)
        afeat=F.sigmoid(self.al(afeat))
        afeat=self.val(afeat)
        return F.pairwise_distance(vfeat, afeat)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        dist = dist.view(-1)
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss
