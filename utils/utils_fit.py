import torch
from tqdm import tqdm

from utils.utils import get_lr, show_result


def fit_one_epoch(G_model_train, D_model_train, G_model, D_model, G_optimizer, D_optimizer, BCE_loss, 
                epoch, epoch_step, gen, Epoch, cuda, batch_size, save_interval):
    G_total_loss = 0
    D_total_loss = 0

    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, images in enumerate(gen):
            if iteration >= epoch_step:
                break
            y_real  = torch.ones(batch_size)
            y_fake  = torch.zeros(batch_size)
            images  = torch.from_numpy(images).type(torch.FloatTensor)

            if cuda:
                images, y_real, y_fake  = images.cuda(), y_real.cuda(), y_fake.cuda()
            
            #----------------------------------------------------#
            #   先训练评价器
            #   利用真假图片训练评价器
            #   目的是让评价器更准确
            #----------------------------------------------------#
            D_optimizer.zero_grad()
            D_result                = D_model_train(images)
            D_real_loss             = BCE_loss(D_result, y_real)
            D_real_loss.backward()

            noise                   = torch.randn((batch_size, 100))
            if cuda:
                noise               = noise.cuda()
            G_result                = G_model_train(noise)
            D_result                = D_model_train(G_result)
            D_fake_loss             = BCE_loss(D_result, y_fake)
            D_fake_loss.backward()
            D_optimizer.step()

            #----------------------------------------------------#
            #   再训练生成器
            #   目的是让生成器生成的图像，被评价器认为是正确的
            #----------------------------------------------------#
            G_optimizer.zero_grad()
            noise                   = torch.randn((batch_size, 100))
            if cuda:
                noise               = noise.cuda()
            G_result                = G_model_train(noise)
            D_result                = D_model_train(G_result).squeeze()
            G_train_loss            = BCE_loss(D_result, y_real)
            G_train_loss.backward()
            G_optimizer.step()
            
            G_total_loss            += G_train_loss
            D_total_loss            += D_real_loss + D_fake_loss

            pbar.set_postfix(**{'G_loss'    : G_total_loss.item() / (iteration + 1), 
                                'D_loss'    : D_total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(G_optimizer)})
            pbar.update(1)

            if iteration % save_interval == 0:
                show_result(epoch + 1, G_model_train, cuda)
                
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss / (epoch_step + 1), D_total_loss / (epoch_step + 1)))
    print('Saving state, iter:', str(epoch + 1))

    #----------------------------#
    #   每10个时代保存一次
    #----------------------------#
    if (epoch + 1) % 10==0:
        torch.save(G_model.state_dict(), 'logs/G_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%(epoch + 1, G_total_loss / (epoch_step + 1), D_total_loss / (epoch_step + 1)))
        torch.save(D_model.state_dict(), 'logs/D_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%(epoch + 1, G_total_loss / (epoch_step + 1), D_total_loss / (epoch_step + 1)))
