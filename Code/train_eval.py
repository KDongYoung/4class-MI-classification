import torch
import torch.nn.functional as F

## TRAIN
def train(log_interval, model, device, train_loader, optimizer, scheduler, alpha, epoch=1):
    print("================= TRAIN Start =================")
    lossfn = torch.nn.CrossEntropyLoss()
    model.train() 
    
    for batch_idx, datas in enumerate(train_loader):
        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)
        if alpha!=0:
            soft_label = F.log_softmax(output, dim=1)
            pred = soft_label.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct=pred.eq(target.view_as(pred)).sum().item()
            cross_entropy_loss = lossfn(output,target) # output은 n행 2열, target은 1행 n열

            pdist=torch.nn.PairwiseDistance(p=2)
            c=soft_label[target==0]
            one_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
            c=soft_label[target==1]
            two_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
            c=soft_label[target==2]
            three_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
            c=soft_label[target==3]
            four_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
            align_loss=one_align_loss+two_align_loss+three_align_loss+four_align_loss

            loss=cross_entropy_loss+alpha*align_loss
        else:
            pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct=pred.eq(target.view_as(pred)).sum().item()
            loss = lossfn(output,target) # output은 n행 2열, target은 1행 n열
        # print(loss)
        loss.backward() # loss backprop
        optimizer.step() # optimizer update parameter

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f} \tACC: {:.2f}'.format(epoch, loss.item(), correct/len(pred)))
        epoch+=1
    scheduler.step() # scheduler update parameter

## EVALUATE
def eval(model, device, test_loader):
    print("================= EVAL Start =================")
    model.eval()  ## nn.Module.eval, evaluation mode로 변경(dropout, batchnorm 사용을 중지함), 다하면 다시 train mode로 변경해줘야 한다.
    # .eval함수와 torch.no_grad함수를 같이 사용하는 경향
    test_loss = []
    correct = []
    lossfn = torch.nn.CrossEntropyLoss()

    preds=[]
    targets=[]
    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
    
            outputs=model(data)
            test_loss.append(lossfn(outputs, target).item()) # sum up batch loss
            pred = outputs.argmax(dim=1,keepdim=True)# get the index of the max probability 인덱스
            correct.append(pred.eq(target.data.view_as(pred)).sum().item())  

            preds.extend(outputs.argmax(dim=1,keepdim=False).cpu().numpy())
            targets.extend(target.cpu().numpy())

    loss = sum(test_loss)/len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'
        .format(loss, sum(correct), len(test_loader.dataset), 100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)

def test_eval(model, device, test_loader):
    print("================= EVAL Start =================")
    model.eval()  ## nn.Module.eval, evaluation mode로 변경(dropout, batchnorm 사용을 중지함), 다하면 다시 train mode로 변경해줘야 한다.
    # .eval함수와 torch.no_grad함수를 같이 사용하는 경향

    preds=[]
    with torch.no_grad():
        for datas in test_loader:
            data = datas[0].to(device)
            outputs=model(data)
            preds.extend(outputs.argmax(dim=1,keepdim=False).cpu().numpy())

    return preds