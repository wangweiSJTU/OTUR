import argparse, os, glob
import torch,pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_unet import _NetG,_NetD
from dataset_dep import DatasetFromHdf5
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from random import randint, seed
import random
import cv2

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet") 
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--noise_sigma", default=50, type=int, help="standard deviation of the Gaussian noise (default: 50)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--trainset", default="../../NYU2/", type=str, help="dataset name")
def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    data_list = glob.glob(opt.trainset+"*.h5")

    print("===> Building model")
    model = _NetG()
    criterion = nn.MSELoss(size_average=True)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)

    print("===> Training")
    MSE =[]
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        mse = 0
        for data_name in data_list:
            train_set = DatasetFromHdf5(data_name)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                batch_size=opt.batchSize, shuffle=True)
            a=train(training_data_loader, G_optimizer, model, criterion, epoch)
            mse += a
        mse = mse / len(data_list)
        MSE.append(format(mse))
        save_checkpoint(model, epoch)

    file = open('./checksample/mse_n2n_'+str(opt.noise_sigma)+str(opt.nEpochs)+'.txt','w')
    for mse in MSE:
        file.write(mse+'\n')
    file.close()

    # psnr = eval_dep(model)
    # print("Final psnr is:",psnr)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, G_optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(G_optimizer, epoch-1)
    mse = []
    for param_group in G_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, G_optimizer.param_groups[0]["lr"]))
    #model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        target = Variable(batch[1])
        # rng_stddev1 = np.random.uniform(0.01, noise_max/255.0,[1,1,1])
        # rng_stddev2 = np.random.uniform(0.01, noise_max/255.0,[1,1,1])
        # noise1 = np.random.normal(size=target.shape) * rng_stddev1
        # noise2 = np.random.normal(size=target.shape) * rng_stddev2
        noise1 = np.random.normal(size=target.shape) * opt.noise_sigma/255.0
        noise2 = np.random.normal(size=target.shape) * opt.noise_sigma/255.0
        noise1=torch.from_numpy(noise1).float()
        noise2=torch.from_numpy(noise2).float()
        if opt.cuda:
            target = target.cuda()
            noise1=noise1.cuda()
            noise2=noise2.cuda()
            input = target+noise1
            target=target+noise2

        # train generator G
        model.zero_grad()

        G_result = model(input)

        mse_loss = (torch.mean((G_result- target)**2))**0.5
        mse.append(mse_loss.data)


        G_train_loss = mse_loss
        G_train_loss.backward()
        G_optimizer.step()
        
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss_mse: {:.5}".format(epoch, iteration, len(training_data_loader), mse_loss.data))
    save_image(G_result.data, './checksample/output.png')
    save_image(input.data, './checksample/input.png')
    save_image(target.data, './checksample/gt.png')


    return torch.mean(torch.FloatTensor(mse))


    return torch.mean(torch.FloatTensor(mse))
   
def save_checkpoint(model, epoch):
    model_out_path = "./checkpoint/model_unet_n2n"+str(opt.noise_sigma)+str(opt.nEpochs)+".pth"
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
