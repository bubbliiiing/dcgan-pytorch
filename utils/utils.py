import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_result(num_epoch, G_net, cuda):
    with torch.no_grad():
        randn_in = torch.randn((5 * 5, 100))
        if cuda:
            randn_in.cuda()

        G_net.eval()
        test_images = G_net(randn_in)
        G_net.train()

        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(test_images[k].cpu().data.numpy().transpose(1, 2, 0) * 0.5 + 0.5)

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig("results/train_out/epoch_" + str(num_epoch) + "_results.png")
        plt.close('all')  #避免内存泄漏

    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
