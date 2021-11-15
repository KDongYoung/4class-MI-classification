import argparse
import random
import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from BCIC4_2A_copy import DataGenerator

# from DCGAN import *
from DCGAN import *

def Experiment(args, subject_id, data_shape):
    # RESULT 생성 DIRECTORY 만들기
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')
        os.makedirs(path + '/Generate')

    with open(path + '/args.txt', 'w') as f:
        f.write(str(args))

    import torch.cuda
    cuda = torch.cuda.is_available()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed) ################## 설마 이 친구의 영향이 클까? 어디서 random이 발생해서 돌릴때마다 다르게 나오냐고?
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Loss function
    lossfn = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = DCGAN_Generator(args, featmap_dim=args.G_featmap_dim, n_channel=data_shape[0], noise_dim=args.latent_dim)
    discriminator = DCGAN_Discriminator(args, featmap_dim=args.D_featmap_dim, n_channel=data_shape[0])

    if cuda:
        generator.cuda()
        discriminator.cuda()
        lossfn.cuda()

    # Configure data loader
    args.mode="train"
    dataloaders = DataGenerator(args, subject_id+1)
    trainloader = dataloaders.train_loader

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    writer = SummaryWriter(f'{path}/S{subject_id+1:02}') # log directory

    ################ Training
    for epoch in range(args.n_epochs):
        for i, (x,y,_) in enumerate(trainloader):

            # Adversarial ground truths
            valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False) #.to(torch.int64)
            fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False) #.to(torch.int64)

            real_data = Variable(x.type(Tensor))
            noise = Variable(Tensor(np.random.normal(0, 1, (x.shape[0], args.latent_dim))))

            # Generate a batch of images
            gen_data = generator(noise)
            ################ Train Generator
            optimizer_G.zero_grad()
            g_loss = lossfn(discriminator(gen_data), valid)
            g_loss.backward()
            optimizer_G.step()

            ################ Train Discriminator
            optimizer_D.zero_grad()
            # inputs = torch.cat([real_data, gen_data.detach()])
            # labels = np.zeros(2 * args.batch_size)
            # labels[:args.batch_size] = 1
            # labels = torch.from_numpy(labels.astype(np.float32))
            # if cuda:
            #     labels = labels.cuda()
            # labels = Variable(labels)
            # d_loss = lossfn(discriminator(inputs)[:,0],labels)

            real_loss = lossfn(discriminator(real_data), valid)
            fake_loss = lossfn(discriminator(gen_data.detach()), fake)
            d_loss = (fake_loss+real_loss)/2 # _3는 /2, _2는 no, 1은 위에 방법
            d_loss.backward()           
            optimizer_D.step()           

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(trainloader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(trainloader) + i
            if batches_done % args.sample_interval == 0:
                if batches_done==0:
                    x_array=gen_data.data.cpu().numpy().squeeze(1)
                    y_array=y
                else:
                    x_array=np.concatenate((x_array,gen_data.data.cpu().numpy().squeeze(1)))
                    y_array=np.concatenate((y_array,y.data.cpu().numpy()))

            writer.add_scalar('total/g_loss', g_loss, batches_done)
            writer.add_scalar('total/d_loss', d_loss, batches_done) 
    writer.close()
    print(x_array.shape, y_array.shape)
    np.save(args.save_root + '/Generate/S0'+str(subject_id+1)+'_X', x_array)
    np.save(args.save_root + '/Generate/S0'+str(subject_id+1)+'_y', y_array)

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default="train", choices=['train', 'test'])
    parser.add_argument('--all_subject', action='store_true')
    parser.add_argument('--get_prediction', default=False) # action='store_true')
    parser.add_argument('--generate', default=False) # action='store_true')
    parser.add_argument('--data-root', default='./bcic4-2a')
    parser.add_argument('--seed', default=2021, help='Seed value')
    parser.add_argument('--save-root', default='./1010_2/') # lr=0.005, wd=0.001, seed=2021
    parser.add_argument('--result-dir', default='/')

    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--d_lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--g_lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=22, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--time_window", type=int, default=500, help="interval")
    parser.add_argument("--sample_interval", type=int, default=98, help="interval betwen image samples")
    parser.add_argument("--D_featmap_dim", type=int, default=64)
    parser.add_argument("--G_featmap_dim", type=int, default=128)

    args = parser.parse_args()
    # print(args)

    data_shape = (args.channels, args.img_size, args.time_window)  
    for id in range(9):
        print("~"*25 + ' Valid Subject ' + str(id+1) + " " + "~"*25)
        Experiment(args, id, data_shape)
