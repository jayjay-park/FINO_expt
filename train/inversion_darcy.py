

from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from train_utils.utils import count_params

torch.manual_seed(0)
np.random.seed(0)


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def get_grid(self, shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)

pretrain = False
finetune = not pretrain

TRAIN_PATH = '../data/darcy_s61_N1200.mat'
TEST_PATH = '../data/darcy_s61_N1200.mat'

ntrain = 1000
ntest = 1

batch_size = 1
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 1
h = int(((127 - 1)/r) + 1)
s = h

print(s)

path = 'PINO_FDM_darcy_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)

'''
Load Model
'''
config = load_config("configs/eigenvectors/e_400.yaml")
ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250617_131205/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=204_val_rel_l2_loss=0.0172.ckpt"

'''
Load dataset
x_test
y_test
'''
# set batch size for inversion
config.data_settings['batch_size'] = 1

dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
dataloader = dataset.get_dataloader(offset=0, limit=1)  # choose sample index with offset

# get one batch
batch = next(iter(dataloader))
x_true = batch['x'].to(device)  # Darcy field
y_true = batch['y'].to(device)  # solution

print("x_true:", x_true.shape, "y_true:", y_true.shape)


grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)


myloss = LpLoss(size_average=False)


def FDM_Darcy(u, a, D=1, f=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)

    return Du


def PINO_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
    index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                         torch.tensor(range(0, size))], dim=0).long()

    boundary_u = u[:, index_x, index_y]
    truth_u = torch.zeros(boundary_u.shape, device=u.device)
    loss_bd = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss(Du, f)


    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f, loss_bd

error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()
grid = grid.cuda()
mollifier = torch.sin(np.pi*grid[...,0]) * torch.sin(np.pi*grid[...,1]) * 0.001

print(mollifier.shape)
# if pretrain:
#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
#                                                shuffle=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
#                                               shuffle=False)

#     model = FNO2d(modes, modes, width).cuda()
#     num_param = count_params(model)
#     print(num_param)
#     optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#     # for ep in range(epochs):
    #     model.train()
    #     t1 = default_timer()
    #     train_pino = 0.0
    #     train_l2 = 0.0
    #     train_loss = 0
    #     for x, y in train_loader:
    #         x, y = x.cuda(), y.cuda()

    #         optimizer.zero_grad()
    #         out = model(x.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
    #         out = out * mollifier

    #         loss_data = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
    #         loss_f, loss_bd = PINO_loss(out, x)
    #         pino_loss = loss_f
    #         pino_loss.backward()

    #         optimizer.step()
    #         train_l2 += loss_data.item()
    #         train_pino += pino_loss.item()
    #         train_loss += torch.tensor([loss_bd, loss_f])

    #     scheduler.step()

    #     model.eval()
    #     test_l2 = 0.0
    #     test_pino = 0.0
    #     with torch.no_grad():
    #         for x, y in test_loader:
    #             x, y = x.cuda(), y.cuda()

    #             out = model(x.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
    #             out = out * mollifier

    #             test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    #             loss_f, loss_bd = PINO_loss(out, x)
    #             test_pino += loss_f.item() + loss_bd.item()

    #     train_l2 /= ntrain
    #     test_l2 /= ntest
    #     train_pino /= ntrain
    #     test_pino /= ntest
    #     train_loss /= ntrain

    #     error[ep] = [train_pino, train_l2, test_pino, test_l2]

    #     t2 = default_timer()
    #     print(ep, t2-t1, train_pino, train_l2, test_pino, test_l2)
    #     print(train_loss)

    # torch.save(model, '../model/IP-dracy-forward')

def darcy_mask1(x):
    return 1 / (1 + torch.exp(-x)) * 0.8 + 0.1

def darcy_mask2(x):
    x = 1 / (1 + torch.exp(-x))
    x[x>0.5] = 1
    x[x<=0.5] = 0
    # x = torch.tensor(x>0.5, dtype=torch.float)
    return  x * 0.8 + 0.1

def total_variance(x):
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))


if finetune:
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    model = torch.load('../model/IP-dracy-forward').cuda()
    num_param = count_params(model)
    print(num_param)
    xout = torch.rand([1,s,s,1], requires_grad=True, device="cuda")

    optimizer = Adam([xout], lr=0.1, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)

    for ep in range(10000):
        model.train()
        t1 = default_timer()

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out_masked = darcy_mask1(xout)

            yout = model(out_masked.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
            yout = yout * mollifier
            loss_data = myloss(yout.view(batch_size, -1), y.view(batch_size, -1))
            loss_f, loss_bd = PINO_loss(y, out_masked)
            loss_TV = total_variance(xout)
            pino_loss = 0.2 * loss_f + loss_data + 0.05 * loss_TV
            # pino_loss = 0. * loss_f + loss_data + 0.05 * loss_TV
            pino_loss.backward()
            optimizer.step()
            scheduler.step()

            out_masked2 = darcy_mask2(xout)
            yout2 = model(out_masked2.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
            yout2 = yout2 * mollifier
            testx_l2 = myloss(out_masked.view(batch_size, -1), x.view(batch_size, -1)).item()
            testy_l2 = myloss(yout.view(batch_size, -1), y.view(batch_size, -1)).item()



        t2 = default_timer()
        print(ep, t2 - t1, loss_data.item(), loss_f.item(), testx_l2, testy_l2)

        if ep % 2000 == 1:
            # fig, axs = plt.subplots(2, 3, figsize=(8, 8))
            # axs[0,0].imshow(x.reshape(s,s).detach().cpu().numpy())
            # axs[0,1].imshow(out_masked.reshape(s,s).detach().cpu().numpy())
            # axs[0,2].imshow(out_masked2.reshape(s,s).detach().cpu().numpy())
            # axs[1,0].imshow(y.reshape(s,s).detach().cpu().numpy())
            # axs[1,1].imshow(yout.reshape(s,s).detach().cpu().numpy())
            # axs[1,2].imshow(yout2.reshape(s,s).detach().cpu().numpy())
            # plt.show()
            name_tag = 'PINO-'
            plt.imshow(x.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'true-input.pdf',bbox_inches='tight')
            plt.imshow(out_masked.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'raw-input.pdf',bbox_inches='tight')
            plt.imshow(out_masked2.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'clip-input.pdf',bbox_inches='tight')

            plt.imshow(y.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'true-output.pdf',bbox_inches='tight')
            plt.imshow(yout.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'raw-output.pdf',bbox_inches='tight')
            plt.imshow(yout.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'clip-output.pdf',bbox_inches='tight')

            # scipy.io.savemat('../pred/IP-darcy-forward.mat', mdict={'input_truth': x.reshape(s,s).detach().cpu().numpy(),
            #                                    'input_pred': out_masked.reshape(s,s).detach().cpu().numpy(),
            #                                     'output_truth': y.reshape(s,s).detach().cpu().numpy(),
            #                                     'output_pred': yout.reshape(s,s).detach().cpu().numpy()})
