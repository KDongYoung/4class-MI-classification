from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np
import os
from BCIC4_2A import *

# models.py, train_eval_shallow.py
from models import ShallowConvNet_ch
from train_eval import *
###########################################################################
n_classes = 4  # class
n_channels=22 # eeg channel
n_frequency=500
###########################################################################

def Experiment(args, subject_id, alpha):
    args.gpuidx= 0

    # RESULT 생성 DIRECTORY 만들기
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')
        os.makedirs(args.save_root + '/Prediction_ShallowConvNet')

    with open(path + '/args.txt', 'w') as f:
        f.write(str(args))

    import torch.cuda
    cuda = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.mode="train"
    dataloaders = DataGenerator(args, subject_id+1)
    train_loader = dataloaders.train_loader
    valid_loader = dataloaders.valid_loader

    model = ShallowConvNet_ch(n_classes, n_channels, n_frequency)
    if cuda:
        model.cuda(device=device) # DEVICE에 연결

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # args.lr
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-1)
    writer = SummaryWriter(f'{path}/S{subject_id+1:02}') # log directory

    """    Training, Validation     """
    best_acc=0
    best_loss=10.0
    for epochidx in range(1, args.epochs):
        print("EPOCH_IDX: ",epochidx)
        train(20, model, device, train_loader, optimizer, scheduler, alpha) # train
        valid_loss, valid_score = eval(model, device, valid_loader) # valid
        
        # compare validation accuracy of this epoch with the best accuracy score
        # if validation accuracy >= best accuracy, then save model(.pt)
        if valid_score >= best_acc:
            print("Higher accuracy then before: epoch {}".format(epochidx))
            best_acc = valid_score
            torch.save(model.state_dict(), os.path.join(path, 'models',"subject{}_bestmodel".format(subject_id+1)))
        writer.add_scalar('total/valid_loss', valid_loss, epochidx)
        writer.add_scalar('total/valid_acc', valid_score, epochidx) 
    writer.close()
    
    # test the best accuracy model
    print("Testing...")
    args.mode="test"
    dataloaders = DataGenerator(args, subject_id+1)
    test_loader = dataloaders.test_loader
    best_model = ShallowConvNet_ch(n_classes, n_channels, n_frequency)
    best_model.load_state_dict(torch.load(os.path.join(path, 'models',"subject{}_bestmodel").format(subject_id+1), map_location=device))
    if cuda: 
        best_model.cuda(device=device)
    pred=test_eval(best_model, device, test_loader)
    # with open(path+'_prediction.txt', 'a') as f:
    #     f.write("subject {}: {}\n".format(subject_id+1, pred))
    np.save(args.save_root + '/Prediction_ShallowConvNet/S0'+str(subject_id+1), pred)
    
    return valid_score

###########################################################################
def main(args, alpha):
    if alpha==0:
        exp_type="ShallowConvNet"
    else:
        exp_type="ShallowConvNet_align"+str(alpha)
        
    args.result_dir=exp_type
    
    path = args.save_root + args.result_dir
    best_model_result_path=path+"/"+exp_type+'_Accuracy.txt'

    vaccs=[]
    for id in range(9):
        print("~"*25 + ' Valid Subject ' + str(id+1) + " " + "~"*25)
        vacc = Experiment(args, id, alpha)

        # the accuracy and loss earned from testing the best accuracy model from train and validation set
        vaccs.append(vacc)

        print(f"TEST SUBJECT ID : S{id+1}, ACCURACY : {vacc:.2f}%")
        with open(best_model_result_path, 'a') as f:
            f.write("subject {}, vacc: {}\n".format(id+1, vacc)) # save test accuracy, test loss, sleep chance level

    print(f"TOTAL AVERAGE : {np.mean(vacc):.2f}%")
    with open(best_model_result_path, 'a') as f:
            f.write("TOTAL AVERAGE: {}\n".format(np.mean(vaccs))) # save mean test accuracy