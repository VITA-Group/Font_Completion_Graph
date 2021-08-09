import os
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from utilities.AverageMeter import AverageMeter
from utilities.accuracy import *
from utilities.EarlyStopping import *


################################################
# This python file contains four parts:
#
# Part 1. Argument Parser
# Part 2. configurations:
#   Part 2-1. Basic configuration
#   Part 2-2. dataloader instantiation
#   Part 2-3. log configuration
#   Part 2-4. configurations for loss function, model, and optimizer
# Part 3. 'train' function
# Part 4. 'validate' function
# Part 5. 'main' function
################################################


class TrainerBase():
    def __init__(self, model, optimizer, logger, writer):
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.writer = writer

    def points2img_8x8(self, points, stroke_len, imgpath, indices=None):
        if indices is not None:
            indices = indices.detach().cpu().numpy()

        fig, ax = plt.subplots(max(len(points) // 8 + int(len(points) % 8 != 0), 2), 8, figsize=(8, 8))

        for n, ps in enumerate(points):
            ps = ps.detach().cpu().numpy()

            if indices is not None:
                idx = np.where(indices[n] != 0)[0]
                ps = ps.take(idx, axis=0)
            else:
                ps = ps[:stroke_len[n]]

            ax[n // 8, n % 8].scatter(ps[:, 0], -ps[:, 1], s=1)
            ax[n // 8, n % 8].set_xlim(-25, 275)
            ax[n // 8, n % 8].set_ylim(-275, 25)
            ax[n // 8, n % 8].axis('off')
            # ax[n // 8, n % 8].axis('equal')

        fig.tight_layout()
        plt.axis('equal')
        plt.savefig(imgpath, bbox_inches='tight')
        plt.close()

    def points2img_9x3(self, points, stroke_len, imgpath, indices=None):
        if indices is not None:
            indices = indices.detach().cpu().numpy()

        fig, ax = plt.subplots(max(len(points) // 9 + int(len(points) % 9 != 0), 2), 9, figsize=(9, 3))

        for n, ps in enumerate(points):
            ps = ps.detach().cpu().numpy()

            if indices is not None:
                idx = np.where(indices[n] != 0)[0]
                ps = ps.take(idx, axis=0)
            else:
                ps = ps[:stroke_len[n]]

            ax[n // 9, n % 9].scatter(ps[:, 0], -ps[:, 1], s=1)
            ax[n // 9, n % 9].set_xlim(-25, 275)
            ax[n // 9, n % 9].set_ylim(-275, 25)
            ax[n // 9, n % 9].axis('off')
            # ax[n // 9, n % 9].axis('equal')

        fig.tight_layout()
        plt.axis('equal')
        plt.savefig(imgpath, bbox_inches='tight')
        plt.close()

    def points2img_row(self, points, stroke_len, imgpath, indices=None):
        if indices is not None:
            indices = indices.detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=26, figsize=(26, 1))

        for n, ps in enumerate(points):
            ps = ps.detach().cpu().numpy()

            if indices is not None:
                idx = np.where(indices[n] != 0)[0]
                ps = ps.take(idx, axis=0)
            else:
                ps = ps[:stroke_len[n]]

            ax[n].scatter(ps[:, 0], -ps[:, 1], s=1)
            ax[n].set_xlim(-25, 275)
            ax[n].set_ylim(-275, 25)
            ax[n].axis('off')
            # ax[n].axis('equal')

        fig.tight_layout()
        plt.axis('equal')
        plt.savefig(imgpath, bbox_inches='tight')
        plt.close()

    def points2img_1xn(self, points, stroke_len, imgpath, indices=None):
        if indices is not None:
            indices = indices.detach().cpu().numpy()

        fig, ax = plt.subplots(1, len(points), squeeze=False)

        for n, ps in enumerate(points):
            ps = ps.detach().cpu().numpy()

            if indices is not None:
                idx = np.where(indices[n] != 0)[0]
                ps = ps.take(idx, axis=0)
            else:
                ps = ps[:stroke_len[n]]

            ax[0, n].scatter(ps[:, 0], -ps[:, 1], s=1)
            ax[0, n].axis('off')
            ax[0, n].axis('equal')
            ax[0, n].set_xlim(-25, 275)
            ax[0, n].set_ylim(-275, 25)

        fig.tight_layout()
        plt.savefig(imgpath, bbox_inches='tight')
        plt.close()

    def _forward(self, data, decode_all=False):
        return {}

    def forward(self, data):
        return None, None, None, None, None

    def set_model_status(self, train=True):
        if train:
            if isinstance(self.model, dict):
                for k in self.model:
                    self.model[k].train()
            else:
                self.model.train()
        else:
            if isinstance(self.model, dict):
                for k in self.model:
                    self.model[k].eval()
            else:
                self.model.eval()

    def save_model(self, save_dir, save_name, epoch):
        checkpoint_path = os.path.join(save_dir, '_'.join([save_name, str(epoch + 1)]) + '.pth')

        if isinstance(self.model, dict):
            model_state = {"epoch": epoch + 1,
                           'optimizer': self.optimizer.state_dict()}
            for k, m in self.model.items():
                model_state[k] = m.state_dict()
            torch.save(model_state, checkpoint_path)

        else:
            model_state = {"epoch": epoch + 1,
                           'optimizer': self.optimizer.state_dict(),
                           "model": self.model.state_dict()}
            torch.save(model_state, checkpoint_path)

    def batch_process(self, data):
        res, label, logit, batch_loss, loss_dict = self.forward(data)
        return label, logit, batch_loss, loss_dict

    def train(self, train_loader, test_loader, epoch, lr):
        training_loss = AverageMeter()
        training_acc = AverageMeter()
        running_loss = {}

        self.set_model_status(train=True)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger.info("set learning rate to: {}".format(lr))

        for idx, data in enumerate(tqdm(train_loader, ascii=True)):
            self.optimizer.zero_grad()

            label, logit, batch_loss, loss_dict = self.batch_process(data)

            if idx == 0:
                for k, v in loss_dict.items():
                    running_loss[k] = AverageMeter()
            for k, v in loss_dict.items():
                running_loss[k].update(batch_loss.item(), label.size(0))

            batch_loss.backward()

            self.optimizer.step()

            training_loss.update(batch_loss.item(), label.size(0))

            training_acc.update(accuracy(logit, label, topk=(1,))[0].item(), label.size(0))

        self.logger.info("average loss: {}   average acc: {}".format(training_loss.avg, training_acc.avg))

        self.logger.info("Begin evaluating on testing set")
        test_loss, test_acc, test_running_loss = self.validate(test_loader)

        self.writer.add_scalar("loss/train", training_loss.avg, epoch + 1)
        self.writer.add_scalar("loss/test", test_loss.avg, epoch + 1)

        self.writer.add_scalar("acc/train", training_acc.avg, epoch + 1)
        self.writer.add_scalar("acc/test", test_acc.avg, epoch + 1)

        if running_loss:
            for k, v in running_loss.items():
                self.writer.add_scalar("loss/%s_train" % k, v.avg, epoch + 1)

        if test_running_loss:
            for k, v in test_running_loss.items():
                self.writer.add_scalar("loss/%s_test" % k, v.avg, epoch + 1)

        return test_acc

    def validate(self, data_loader):
        validation_loss = AverageMeter()
        running_loss = {}

        validation_acc_1 = AverageMeter()
        validation_acc_5 = AverageMeter()
        validation_acc_10 = AverageMeter()

        self.set_model_status(train=False)

        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, ascii=True)):
                label, logit, batch_loss, loss_dict = self.batch_process(data)

                if idx == 0:
                    for k, v in loss_dict.items():
                        running_loss[k] = AverageMeter()
                for k, v in loss_dict.items():
                    running_loss[k].update(batch_loss.item(), label.size(0))

                validation_loss.update(batch_loss.item(), label.size(0))

                acc_1, acc_5, acc_10 = accuracy(logit, label, topk=(1, 5, 10))
                validation_acc_1.update(acc_1, label.size(0))
                validation_acc_5.update(acc_5, label.size(0))
                validation_acc_10.update(acc_10, label.size(0))

            self.logger.info("loss: {}  acc@1: {} acc@5: {} acc@10: {}".format(
                validation_loss.avg, validation_acc_1.avg, validation_acc_5.avg, validation_acc_10.avg))

        self.logger.info("loss: {}".format(running_loss.keys()))
        return validation_loss, validation_acc_1, running_loss

    def stream(self, train_loader, test_loader, lr_protocol, patience,
               startpoint, num_epochs, save_dir, save_name, min_epoch=200):
        max_val_acc = 0.0
        max_val_acc_epoch = -1

        self.logger.info("Evaluating on validation set before training")
        self.validate(test_loader)

        self.logger.info("training status: ")
        early_stopping = EarlyStopping(patience=patience, delta=0)

        for epoch in range(startpoint, num_epochs):
            self.logger.info("Begin training epoch {}".format(epoch + 1))

            lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch > epoch), lr_protocol[-1][1])
            validation_acc = self.train(train_loader, test_loader, epoch, lr)

            if validation_acc.avg > max_val_acc:
                max_val_acc = validation_acc.avg
                max_val_acc_epoch = epoch + 1

            early_stopping(validation_acc.avg)
            self.logger.info("Early stopping counter: {}".format(early_stopping.counter))
            self.logger.info("Early stopping best_score: {}".format(early_stopping.best_score))
            self.logger.info("Early stopping early_stop: {}".format(early_stopping.early_stop))

            if early_stopping.early_stop == True and epoch >= min_epoch:
                self.logger.info("Early stopping after Epoch: {}".format(epoch + 1))
                break

            if (epoch + 1) % 5 == 0:
                self.save_model(save_dir, save_name, epoch)

        self.logger.info("max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch))
