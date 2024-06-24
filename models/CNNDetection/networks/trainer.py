import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel
from models.DANN.adversarialNetwork import AdversarialNetwork


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.isDANN = opt.train.DANN if hasattr(opt.train, 'DANN') else False
        if self.isTrain and not opt.train.continue_train:
            self.model = resnet50(pretrained=True, DANN_output=self.isDANN)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

        if not self.isTrain or opt.train.continue_train:
            self.model = resnet50(num_classes=1, DANN_output=self.isDANN)

        if self.isDANN:
            self.adversarial = AdversarialNetwork(in_feature=2048) # 2048 is the output of the resnet50 before the fc layer
            self.source_gens = opt.train.DANN_config.source
            self.target_gens = opt.train.DANN_config.target

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
            
            if self.isDANN:
                self.optimizer.add_param_group({'params': self.adversarial.parameters(), 'lr': opt.train.lr})
                
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
        if self.isDANN:
            self.adversarial.to(self.device)


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        self.generator_label = input[2].to(self.device).float()


    def forward(self, inputs=None):
        if inputs is not None:
            self.input = inputs.to(self.device)
        # else inputs are already set in set_input

        if self.isDANN:
            features, self.output = self.model(self.input)
            self.DANN_output = self.adversarial(features)
            return features, self.output
        else:
            self.output = self.model(self.input)
            return self.output

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        if self.isDANN:
            # filter out non-source generators
            indices = [i for i in range(len(self.generator_label)) if self.generator_label[i] in self.source_gens]
            source_outputs = self.output[indices]
        else:
            source_outputs = self.output
            indices = []*len(self.generator_label)
        self.classify_loss = self.loss_fn(source_outputs.squeeze(1), self.label[indices])
        self.loss = self.classify_loss

        if self.isDANN:
            self.DANN_loss = self.loss_fn(self.DANN_output.squeeze(1), self.generator_label)
            self.loss += self.DANN_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return {"output": self.output, "loss": self.loss, "DANN_loss": self.DANN_loss if self.isDANN else None, "classify_loss": self.classify_loss}
