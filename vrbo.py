import copy

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from .RNN_net import RNN, NLIRNN
GLOVE_DIM=300
class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, training_size):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.training_size = training_size
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.data == 'snli':
            self.inner_model = NLIRNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
            self.inner_model_old = copy.deepcopy(self.inner_model).cuda()
        self.lambda_x =  torch.ones((self.training_size)).to(self.device)
        self.lambda_x_old =  torch.ones((self.training_size)).to(self.device)
        self.lambda_x.requires_grad=True
        self.lambda_x_old.requires_grad=True
        self.hypermomentum_x =  torch.ones((self.training_size)).to(self.device)
        self.hypermomentum_x_old = torch.ones((self.training_size)).to(self.device)
        self.hypermomentum_y =  [torch.zeros(param.size()).to(self.device) for param in
                                       self.inner_model.parameters()]
        self.hypermomentum_y_old = [torch.zeros(param.size()).to(self.device) for param in
                                       self.inner_model.parameters()]
        self.outer_optimizer = SGD([self.lambda_x], lr=self.outer_update_lr)
        self.inner_optimizer = SGD(self.inner_model.parameters(), lr=self.inner_update_lr)
        self.inner_model.train()
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training = True, epoch = 0):
        # self.model.load_state_dict(torch.load('checkpoints/itd-model.pkl'))
        task_accs = []
        task_loss = []

        for step, data in enumerate(train_loader):
            self.inner_model.to(self.device)
            self.inner_model_old.to(self.device)
            if step % self.args.update_interval == 0:
                input_val, label_id_val, data_indx_val = next(iter(val_loader))
                outputs_val = predict(self.inner_model, input_val)
                outer_loss = torch.mean(self.criterion(outputs_val, label_id_val.to(self.device)))
                grad_x = self.stocbio(self.args, outer_loss, next(iter(val_loader)), next(iter(val_loader)))
                self.hypermomentum_x = grad_x[0].data
                self.hypermomentum_x_old.data = grad_x[0].data
                self.lambda_x_old.data = self.lambda_x.data.detach()
                self.outer_optimizer.zero_grad()
                self.inner_optimizer.zero_grad()
                input_val, label_id_val, data_indx_val = data
                outputs = predict(self.inner_model, input_val)
                loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx_val])*self.criterion(outputs, label_id_val.to(self.device))) + 0.0001 * sum(
                        [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model.parameters()])
                grad_y = torch.autograd.grad(loss, self.inner_model.parameters())
                for i, g_y in enumerate(grad_y):
                    self.hypermomentum_y[i].data = g_y
                    self.hypermomentum_y_old[i].data = g_y

            self.lambda_x.grad = self.hypermomentum_x
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            self.inner_optimizer.zero_grad()
            self.lambda_x_old.data = self.lambda_x.data.detach()
            self.inner_model_old = copy.deepcopy(self.inner_model)
            for i in range(self.args.spider_loops):
                input_val, label_id_val, data_indx_val = data
                outputs_val = predict(self.inner_model, input_val)
                outer_loss = torch.mean(self.criterion(outputs_val, label_id_val.to(self.device)))
                train_batch1, train_batch2 = next(iter(train_loader)), next(iter(train_loader))
                grad_x = self.stocbio(self.args, outer_loss, train_batch1, train_batch2)[0]

                outputs_val_on_old_model = predict(self.inner_model_old, input_val)
                outer_loss_old = torch.mean(self.criterion(outputs_val_on_old_model, label_id_val.to(self.device)))
                grad_x_on_old_model = self.stocbio_old(self.args, outer_loss_old, train_batch1, train_batch2)[0]
                for i, (gx, gxo) in enumerate(zip(grad_x, grad_x_on_old_model)):
                    temp_hm = self.hypermomentum_x[i].clone()
                    self.hypermomentum_x[i] = self.hypermomentum_x_old[i] +  (gx.detach() - gxo.detach())
                    self.hypermomentum_x_old[i].data =  temp_hm

                input, label_id, data_indx = data
                outputs = predict(self.inner_model, input)
                loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx]) * self.criterion(outputs, label_id.to(
                    self.device))) + 0.0001 * sum(
                    [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model.parameters()])
                grad_y = torch.autograd.grad(loss, self.inner_model.parameters())
                outputs = predict(self.inner_model_old, input)
                loss = torch.mean(torch.sigmoid(self.lambda_x_old[data_indx]) * self.criterion(outputs, label_id.to(
                    self.device))) + 0.0001 * sum(
                    [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model_old.parameters()])
                grad_y_on_old_model = torch.autograd.grad(loss, self.inner_model_old.parameters())
                for i, (gy, gyo) in enumerate(zip(grad_y, grad_y_on_old_model)):
                    temp_hm = self.hypermomentum_y[i].clone()
                    self.hypermomentum_y[i] = self.hypermomentum_y_old[i] +  (gy.detach() - gyo.detach())
                    self.hypermomentum_y_old[i].data =  temp_hm
                self.inner_model_old = copy.deepcopy(self.inner_model)
                for i, p in enumerate(self.inner_model.parameters()):
                    p.grad = self.hypermomentum_y[i]
                self.inner_optimizer.step()
                self.inner_optimizer.zero_grad()

            q_logits = F.softmax(outputs_val, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = label_id_val.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(outer_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')

        return np.mean(task_accs), np.mean(task_loss)

    def collate_pad_(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0])==list else  data_points[1]
        targets = data_points[1] if type(data_points[0])==list else  data_points[0]

        # Get sentences for batch and their lengths.
        s_lens = np.array([sent.shape[0] for sent in s_embeds])
        max_s_len = np.max(s_lens)
        # Encode sentences as glove vectors.
        bs = len(data_points[0])
        s_embed = np.zeros((max_s_len, bs, GLOVE_DIM))
        for i in range(bs):
            e = s_embeds[i]
            if len(e) <= 0:
                s_lens[i] = 1
            s_embed[: len(e), i] = e.copy()
        embeds = torch.from_numpy(s_embed).float().to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return (embeds, s_lens), targets

    def test(self, test_loader):
        task_accs = []
        task_loss = []

        self.inner_model.to(self.device)
        for step, data in enumerate(test_loader):
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            q_loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))

            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        return np.mean(task_accs), np.mean(task_loss)

    def stocbio(self, args, loss, query_batch, support_batch):
        train_data, train_labels, train_indx = support_batch
        val_data, val_labels, val_indx = query_batch
        Fy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), retain_graph=True)
        F_gradient = [g_param.view(-1) for g_param in Fy_gradient]
        v_0 = torch.unsqueeze(torch.reshape(torch.hstack(F_gradient), [-1]), 1).detach()
        # Hessian
        z_list = []
        outputs = predict(self.inner_model, train_data)
        loss = torch.mean(torch.sigmoid(self.lambda_x[train_indx])*self.criterion(outputs, train_labels.to(self.device))) + 0.0001 * sum(
                [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model.parameters()])
        G_gradient = []
        Gy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), create_graph=True)

        for g_grad, param in zip(Gy_gradient, self.inner_model.parameters()):
            G_gradient.append((param - args.neumann_lr * g_grad).view(-1))
        G_gradient = torch.reshape(torch.hstack(G_gradient), [-1])

        for _ in range(args.hessian_q):
            Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(Jacobian, self.inner_model.parameters(), retain_graph=True)
            v_params = [v_param.view(-1) for v_param in v_new]
            v_0 = torch.unsqueeze(torch.reshape(torch.hstack(v_params), [-1]), 1).detach()
            z_list.append(v_0)
        v_Q = args.neumann_lr * torch.sum(torch.stack(z_list), dim=0)

        outputs = predict(self.inner_model, val_data)
        loss = torch.mean(torch.sigmoid(self.lambda_x[val_indx])*self.criterion(outputs, val_labels.to(self.device))) + 0.0001 * sum(
                [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model.parameters()])
        Gy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), retain_graph=True, create_graph=True)
        Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
        Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
        Gyxv_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, v_Q.detach()), self.lambda_x)
        outer_update =  Gyxv_gradient

        return outer_update


    def stocbio_old(self, args, loss, query_batch, support_batch):
        train_data, train_labels, train_indx = support_batch
        val_data, val_labels, val_indx = query_batch
        Fy_gradient = torch.autograd.grad(loss, self.inner_model_old.parameters(), retain_graph=True)
        F_gradient = [g_param.view(-1) for g_param in Fy_gradient]
        v_0 = torch.unsqueeze(torch.reshape(torch.hstack(F_gradient), [-1]), 1).detach()

        z_list = []
        outputs = predict(self.inner_model_old, train_data)
        loss = torch.mean(torch.sigmoid(self.lambda_x_old[train_indx])*self.criterion(outputs, train_labels.to(self.device))) + 0.0001 * sum(
                [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model_old.parameters()])
        G_gradient = []
        Gy_gradient = torch.autograd.grad(loss, self.inner_model_old.parameters(), create_graph=True)

        for g_grad, param in zip(Gy_gradient, self.inner_model_old.parameters()):
            G_gradient.append((param - args.neumann_lr * g_grad).view(-1))
        G_gradient = torch.reshape(torch.hstack(G_gradient), [-1])

        for _ in range(args.hessian_q):
            Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(Jacobian, self.inner_model_old.parameters(), retain_graph=True)
            v_params = [v_param.view(-1) for v_param in v_new]
            v_0 = torch.unsqueeze(torch.reshape(torch.hstack(v_params), [-1]), 1).detach()
            z_list.append(v_0)
        v_Q = args.neumann_lr * torch.sum(torch.stack(z_list), dim=0)

        outputs = predict(self.inner_model_old, val_data)
        loss = torch.mean(torch.sigmoid(self.lambda_x_old[val_indx])*self.criterion(outputs, val_labels.to(self.device))) + 0.0001 * sum(
                [x.norm(self.args.p_norm).pow(self.args.p_norm) for x in self.inner_model_old.parameters()])
        Gy_gradient = torch.autograd.grad(loss, self.inner_model_old.parameters(), retain_graph=True, create_graph=True)
        Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
        Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
        Gyxv_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, v_Q.detach()), self.lambda_x_old)
        outer_update =  Gyxv_gradient

        return outer_update

def predict(net, inputs):
    """ Get predictions for a single batch. """
    (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    outputs = net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))
    return outputs


