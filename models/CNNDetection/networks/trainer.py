import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.train.continue_train:
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

        if not self.isTrain or opt.train.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.train.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.train.lr, betas=(opt.train.beta1, 0.999))
            elif opt.train.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.train.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")
        
        # lr scheduler
        if not hasattr(opt.train, 'lr_policy') or opt.train.lr_policy == 'constant':
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)
        elif opt.train.lr_policy == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.train.lr_decay_epoch, gamma=0.1)
        elif opt.train.lr_policy == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=opt.train.lr_decay_epoch, verbose=True)
        else:
            raise ValueError("Unknown lr policy")

        if not self.isTrain or opt.train.continue_train:
            self.load_networks(opt.train.epoch)

        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model.to(self.device)


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        return self.output

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return {"output": self.output, "loss": self.loss}


