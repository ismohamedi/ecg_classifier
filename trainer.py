import os
import os.path as osp
from datetime import datetime
from dataset2d import EcgDataset2D
import numpy as np
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import  ResNet34
from utils.network_utility import load_checkpoint, save_checkpoint

class Trainer:
    def __init__(self):
        self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = osp.join("experiments", self.exp_name, 'logs')
        self.pth_dir = osp.join("experiments", self.exp_name, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pth_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.model = self._init_net()
        self.optimizer = self._init_optimizer()
        self.criterion = nn.CrossEntropyLoss().to("cpu")
        self.train_loader, self.val_loader = self._init_dataloaders()

        self.training_epoch = 0
        self.total_iter = 0
        self.epochs = int(1e5)

    def _init_net(self):
        model = ResNet34()
        model = model.to("cpu")
        return model

    def _init_dataloaders(self):
        train = "data/train.json"
        val = "data/val.json"
        mapping = "data/class-mapper.json"
        train_dl = EcgDataset2D(train, mapping).get_dataloader()
        val_dl = EcgDataset2D(val, mapping).get_dataloader()
        return train_dl, val_dl
    def _init_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        gt_class = np.empty(0)
        pd_class = np.empty(0)

        for i, batch in enumerate(self.train_loader):
            inputs = batch['image'].to("cpu")
            targets = batch['class'].to("cpu")

            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)

            classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

            gt_class = np.concatenate((gt_class, batch['class'].numpy()))
            pd_class = np.concatenate((pd_class, classes))

            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                print("\tIter [%d/%d] Loss: %.4f" %
                      (i + 1, len(self.train_loader), loss.item()))

            self.writer.add_scalar(
                "Train loss (iterations)", loss.item(), self.total_iter)
            self.total_iter += 1
        total_loss /= len(self.train_loader)
        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print('Train loss - {:4f}'.format(total_loss))
        print('Train CLASS accuracy - {:4f}'.format(class_accuracy))

        self.writer.add_scalar('Train loss (epochs)',
                               total_loss, self.training_epoch)
        self.writer.add_scalar('Train CLASS accuracy',
                               class_accuracy, self.training_epoch)

    def val(self):
        self.model.eval()
        total_loss = 0

        gt_class = np.empty(0)
        pd_class = np.empty(0)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_loader)):
                inputs = batch['image'].to("cpu")
                targets = batch['class'].to("cpu")
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()
                gt_class = np.concatenate((gt_class, batch['class'].numpy()))
                pd_class = np.concatenate((pd_class, classes))
                total_loss += loss.item()
        total_loss /= len(self.val_loader)
        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print('Validation loss - {:4f}'.format(total_loss))
        print('Validation CLASS accuracy - {:4f}'.format(class_accuracy))

        self.writer.add_scalar(
            'Validation loss', total_loss, self.training_epoch)
        self.writer.add_scalar('Validation CLASS accuracy',
                               class_accuracy, self.training_epoch)

    def loop(self):
        for epoch in range(self.training_epoch, self.epochs):
            print("Epoch - {}".format(self.training_epoch + 1))
            self.train_epoch()
            save_checkpoint({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'total_iter': self.total_iter
            }, osp.join(self.pth_dir, '{:0>8}.pth'.format(epoch)))
            self.val()
            self.training_epoch += 1

if __name__ == "__main__":
    trainer = Trainer()
    trainer.loop()
