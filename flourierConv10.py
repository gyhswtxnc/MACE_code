#add both pos amplifier and neg amplifier, no pseudo inverse
#use gamma for each convolution layer in encoder and decoder of autoencoder
#reverse autoencoder
#fuse the pos and negative ampifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from omni_anomaly.eval_methods import pot_eval,searchThreshold
from scipy.stats import norm
import math
import random
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class myDataset(Dataset):
    def __init__(self,datas,Masks,basisR,basisI,parameters,type="train",labels=-1):
        super(myDataset, self).__init__()
        self.type=type
        if type=="train":
            self.x,self.mask,self.basisR,self.basisI,self.Indexes=self.split(datas,Masks,basisR,basisI,parameters.winLen)
        else:
            self.x,self.y,self.mask,self.basisR,self.basisI=self.splitTest(datas,Masks,basisR,basisI,labels,parameters.winLen)
        #print(self.x.shape)

    def split(self,datas,Masks,basisRs,basisIs,winLen):
        x=[]
        mask=[]
        basisReal=[]
        basisImag=[]
        Indexes=[]
        Idx=0
        for data,Mask,basisR,basisI in zip(datas,Masks,basisRs,basisIs):
            for i in range(0,len(data)-winLen):
                x.append(data[i:i+winLen])
                mask.append(Mask)
                basisReal.append(basisR)
                basisImag.append(basisI)
                Indexes.append(Idx)
            Idx+=1
        x=np.array(x,np.float)
        x=torch.tensor(x,dtype=torch.float)
        x=x.transpose(-1,-2)
        mask=torch.stack(mask,dim=0)
        basisReal=torch.stack(basisReal,dim=0)
        basisImag=torch.stack(basisImag,dim=0)
        Indexes=torch.tensor(Indexes)
        print(x.shape)
        #x=x.permute((0,2,1))
        #x=torch.permute(x,(0,2,1))
        return x,mask,basisReal,basisImag,Indexes

    def splitTest(self,datas,Masks,basisRs,basisIs,labels,winLen):
        x = []
        y=[]
        mask=[]
        basisReal=[]
        basisImag=[]
        for data,label,Mask,basisR,basisI in zip(datas,labels,Masks,basisRs,basisIs):
            if len(data)!=len(label):
                print("data label not match!")
                exit(0)
            for i in range(0,len(data) - winLen):
                x.append(data[i:i + winLen])
                y.append(label[i:i+winLen])
                mask.append(Mask)
                basisReal.append(basisR)
                basisImag.append(basisI)
        x=np.array(x,np.float)
        x = torch.tensor(x,dtype=torch.float)
        x=x.transpose(-1,-2)
        #x=x.permute((0,2,1))
        #x = torch.permute(x, (0, 2, 1))
        y=np.array(y,np.float)
        y=torch.tensor(y,dtype=torch.float)
        masks=torch.stack(mask,dim=0)
        basisReals=torch.stack(basisReal,dim=0)
        basisImags=torch.stack(basisImag,dim=0)
        print(x.shape)
        return x,y,masks,basisReals,basisImags

    def __getitem__(self, item):
        item=item%self.__len__()
        if self.type=="train":
            return (self.x[item],self.mask[item],self.basisR[item],self.basisI[item],self.Indexes[item])
        else:
            return (self.x[item],self.y[item],self.mask[item],self.basisR[item],self.basisI[item])

    def __len__(self):
        return len(self.x)

class myNewTrset(Dataset):
    def __init__(self,x,Masks,basisR,basisI,Indexs,type="train"):
        super(myNewTrset, self).__init__()
        self.type=type
        self.x,self.mask,self.basisR,self.basisI,self.Indexes=x,Masks,basisR,basisI,Indexs

    def __getitem__(self, item):
        item=item%self.__len__()
        return (self.x[item],self.mask[item],self.basisR[item],self.basisI[item],self.Indexes[item])

    def __len__(self):
        return len(self.x)

class ConvAutoEncoder(nn.Module):
    def __init__(self,parameters,ks,ops,isUp):
        super(ConvAutoEncoder, self).__init__()
        self.encodeCelllayers = [nn.Conv1d(3 * parameters.feaLen, 1 * parameters.feaLen, ks[parameters.l - 1], 2)]
        self.decoderCelllayers = []
        self.l = parameters.l
        self.gamma=parameters.gamma
        self.sigma=parameters.sigma
        self.isUp=isUp
        if isUp:
            self.gammaSign=1.
            self.padding=0.
            self.rgammaSign=1.
            self.rpadding=0.
        else:
            self.gammaSign=-1.
            self.padding=0.0001
            self.rgammaSign=-1.
            self.rpadding=0.0001
        for i in range(parameters.l - 2, -1, -1):
            self.encodeCelllayers.append(nn.Conv1d(parameters.feaLen,parameters.feaLen, ks[i], 2))
        for i in range(0, parameters.l, 1):
            self.decoderCelllayers.append(nn.ConvTranspose1d(parameters.feaLen, parameters.feaLen, ks[i], 2,
                                                             output_padding=ops[i]))
        self.encodeCell = nn.ModuleList(self.encodeCelllayers)
        self.decodeCell = nn.ModuleList(self.decoderCelllayers)
        self.batchNorm=nn.BatchNorm1d(parameters.feaLen)

    def forward(self,x):
        last=x
        for i in range(self.l):
            last = torch.pow(torch.pow(last, self.gamma) + self.padding, self.gammaSign) / self.sigma
            lastR=last.real
            lastI=last.imag
            lastR=self.encodeCell[i](lastR)
            lastI=self.encodeCell[i](lastI)
            last=torch.complex(lastR,lastI)
            last=torch.pow(torch.pow(last, 1/self.gamma) + self.padding, self.gammaSign)
            """lastR = last.real
            lastI = last.imag
            lastR = self.batchNorm(lastR)
            lastI = self.batchNorm(lastI)
            last=torch.complex(lastR,lastI)"""
        if torch.isnan(last).any():
            print("encoder cause NAN, up?",self.isUp)
        for i in range(self.l):
            """lastR=last.real
            lastI=last.imag
            lastR = self.batchNorm(lastR)
            lastI = self.batchNorm(lastI)
            last=torch.complex(lastR,lastI)"""
            last = torch.pow(torch.pow(last, self.gamma) + self.rpadding, self.rgammaSign) / self.sigma
            lastR=last.real
            lastI=last.imag
            lastR=self.decodeCell[i](lastR)
            lastI=self.decodeCell[i](lastI)
            last=torch.complex(lastR,lastI)
            last = torch.pow(torch.pow(last, 1 / self.gamma) + self.rpadding, self.rgammaSign)
            #last=self.batchNorm(last)
        if torch.isnan(last).any():
            print("decoder cause NAN, up?",self.isUp)
        return last


class flourierConv(nn.Module):
    def __init__(self,parameters,ks,ops):#ks=[3,4,8,16]
        super(flourierConv,self).__init__()
        self.feaLen=parameters.feaLen
        self.basisNum=parameters.basisNum
        self.gamma=parameters.gamma
        self.sigma=parameters.sigma
        self.device=parameters.device
        self.weightComput=nn.Softmax(dim=-1)
        self.upAuto=ConvAutoEncoder(parameters,ks,ops,True)
        self.loAuto=ConvAutoEncoder(parameters,ks,ops,False)

    def forward(self,x,mask,basisR,basisI):#x:batch,fealen,winlen
        x = torch.fft.fft(x, dim=-1)
        nmask = mask == False
        x[nmask]=0.
        xupper=x.clone()
        xlower=x.clone()
        batch,_,_=x.shape
        xemb=x[mask].reshape(batch,self.feaLen,self.basisNum)
        xemb=torch.cat([basisR,basisI,xemb],dim=1)
        reconsUp=self.upAuto(xemb)
        reconsLo=self.loAuto(xemb)
        #print("recons from auto:",torch.isnan(reconsUpR).any())
        #W=torch.matmul(torch.linalg.pinv(xemb),x)
        #print("psudo inverse:",torch.isnan(W).any())
        xupper[mask]=reconsUp.reshape(-1)
        xlower[mask]=reconsLo.reshape(-1)
        # print("matmul:",torch.isnan(xupper).any())
        """upmask = xupper < 0.
        upSign = torch.ones(upmask.shape).to(self.device)
        upSign[upmask] = -1.
        lomask = xlower < 0.
        loSign = torch.ones(xlower.shape).to(self.device)
        loSign[lomask] = -1."""
        #xupper = torch.pow(xupper, 1 / self.gamma)#xupper = torch.pow(torch.abs(xupper), 1 / self.gamma) * upSign
        #xlower = torch.pow(torch.pow(xlower, 1 / self.gamma) + 0.0001, -1)#xlower = torch.pow(torch.pow(torch.abs(xlower), 1 / self.gamma) + 0.0001, -1) * loSign
        reconsUp=torch.fft.ifft(xupper,dim=-1).real
        reconsLo=torch.fft.ifft(xlower,dim=-1).real
        #print(torch.isnan(reconsUp).any())
        #print()
        return reconsUp,reconsLo

class flourierConAmplified(nn.Module):
    def __init__(self,parameters,ks,ops):
        super(flourierConAmplified, self).__init__()
        self.tgamma = parameters.tgamma
        self.tsigma = parameters.tsigma
        self.timeConvup = nn.Conv1d(parameters.feaLen, parameters.feaLen, parameters.timek, 1,
                                    padding=int((parameters.timek - 1) / 2))
        self.timeConvlo = nn.Conv1d(parameters.feaLen, parameters.feaLen, parameters.timek, 1,
                                    padding=int((parameters.timek - 1) / 2))
        self.flourierConvUp = flourierConv(parameters,ks,ops)
        self.flourierConvLo = flourierConv(parameters,ks,ops)
        self.layerNorm=nn.LayerNorm(parameters.winLen)
        self.device=parameters.device

    def posdePow(self,x,gamma,padding=0.0001,gammaSign=-1):
        xmask=x<0.
        xSign=torch.ones(xmask.shape).to(self.device)
        xSign[xmask]=-1.
        x=torch.pow(torch.pow(torch.abs(x),1/gamma)+padding,gammaSign) * xSign
        return x

    def forward(self,x,mask,basisR,basisI):
        xLo = torch.pow(torch.pow(x, self.tgamma) + 0.0001, -1) / self.tsigma
        if torch.isnan(xLo).any() or torch.isinf(xLo).any():
            print("power in xLo time domain cause NAN",torch.isnan(xLo).any(),torch.isinf(xLo).any())
        xLo = self.timeConvlo(xLo)
        if torch.isnan(xLo).any():
            print("conv in xLo time domain cuase NAN")
        xLo = self.posdePow(xLo, self.tgamma)
        #xLo=self.layerNorm(xLo)
        xUp = torch.pow(x, self.tgamma) / self.tsigma
        if torch.isnan(xUp).any() or torch.isinf(xUp).any():
            print("power in xUp time domain cause NAN",torch.isnan(xUp).any(),torch.isinf(xUp).any())
        xUp = self.timeConvup(xUp)
        if torch.isnan(xUp).any():
            print("conv in xUp time domain cause NAN")
        xUp = self.posdePow(xUp, self.tgamma, 0., 1)
        #xUp=self.layerNorm(xUp)
        x=(xUp+xLo)/2
        if torch.isnan(x).any():
            print("amplified in time domain cause NAN (xLo,xUp):",torch.isnan(xLo).any(),torch.isnan(xUp).any())
        UpU,UpL=self.flourierConvUp(x,mask,basisR,basisI)
        #LoU,LoL=self.flourierConvLo(xLo,mask,basisR,basisI)
        return UpU,UpL


def train(dataloader,model,loss_fn,parameters,optimizer):#optimizer!
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,mask,basisR,basisI,Indexes) in enumerate(dataloader):
        x = x.to(parameters.device)
        mask= mask.to(parameters.device)
        basisR=basisR.to(parameters.device)
        basisI=basisI.to(parameters.device)
        # Compute prediction error
        UpU,UpL = model(x,mask,basisR,basisI)
        loss = loss_fn(UpU,x)+loss_fn(UpL,x)
        # loss=loss_fn(pred,y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2)
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(x)
            # plot(model,X,y,pLen)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def getAnomScore1(x,UpU,UpL):
    upScore1=(torch.pow(x-UpU,2).sum(dim=-2))[:,-1]
    loScore1=(torch.pow(x-UpL,2).sum(dim=-2))[:,-1]
    return torch.max(upScore1,loScore1)


def validate(dataloaders,model,loss_fn,parameters):
    test_losses=[]
    precisions=[]
    recalls=[]
    F1s=[]
    accuracys=[]
    thresholds=[]
    for dataloader in dataloaders:
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        scores = []
        labels = []
        with torch.no_grad():
            for x, y, mask, basisR, basisI in dataloader:
                x, y = x.to(parameters.device), y.to(parameters.device)
                mask = mask.to(parameters.device)
                basisR = basisR.to(parameters.device)
                basisI = basisI.to(parameters.device)
                y = y == 1
                labels.append(y[:, -1])
                UpU, UpL = model(x, mask, basisR, basisI)
                score = getAnomScore1(x, UpU, UpL)
                # torch.true_divide(torch.abs(x-xu),xstd).sum(dim=(-1))
                scores.append(score)
                test_loss += loss_fn(UpU, x).item() + loss_fn(UpL, x).item()
        test_loss /= num_batches
        scores = torch.cat(scores, dim=0).cpu().detach().numpy()
        labels = torch.cat(labels, dim=0).cpu().detach().numpy()
        pot_result = searchThreshold(scores, labels)
        precision = pot_result['pot-precision']
        recall = pot_result['pot-recall']
        F1 = pot_result['pot-f1']
        threshold=pot_result['pot-threshold']
        accuracy = torch.true_divide(pot_result['pot-TP'] + pot_result['pot-TN'],
                                     pot_result['pot-TP'] + pot_result['pot-TN']
                                     + pot_result['pot-FP'] + pot_result['pot-FN']).item()
        test_losses.append(test_loss)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        accuracys.append(accuracy)
        thresholds.append(threshold)
    F1s=np.mean(F1s)
    return test_losses, precisions, recalls, F1s, accuracys, thresholds, scores

def skimming(dataloader,model,parameters,thresholds):
    model.eval()
    nx=[]
    nmask=[]
    nbasisR=[]
    nbasisI=[]
    nindex=[]
    with torch.no_grad():
        for x, mask, basisR, basisI,index in dataloader:
            x = x.to(parameters.device)
            mask = mask.to(parameters.device)
            basisR = basisR.to(parameters.device)
            basisI = basisI.to(parameters.device)
            UpU, UpL = model(x, mask, basisR, basisI)
            score = getAnomScore1(x, UpU, UpL)
            # torch.true_divide(torch.abs(x-xu),xstd).sum(dim=(-1))
            for sc,ind,xi,mi,bri,bii in zip(score,index,x,mask,basisR,basisI):
                if sc<thresholds[ind]:
                    nx.append(xi)
                    nmask.append(mi)
                    nbasisR.append(bri)
                    nbasisI.append(bii)
                    nindex.append(ind)
    nx=torch.stack(nx,dim=0)
    nmask=torch.stack(nmask,dim=0)
    nbasisR=torch.stack(nbasisR,dim=0)
    nbasisI=torch.stack(nbasisI,dim=0)
    nindex=torch.tensor(nindex)
    print(nx.shape,nmask.shape,nbasisR.shape,nbasisI.shape,nindex.shape)
    return nx,nmask,nbasisR,nbasisI,nindex

def test(dataloaders,model,loss_fn,parameters,valscores):
    test_losses = []
    precisions = []
    recalls = []
    F1s = []
    accuracys = []
    for dataloader in dataloaders:
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        scores=[]
        labels=[]
        with torch.no_grad():
            for x, y,mask,basisR,basisI in dataloader:
                x, y = x.to(parameters.device), y.to(parameters.device)
                mask = mask.to(parameters.device)
                basisR = basisR.to(parameters.device)
                basisI = basisI.to(parameters.device)
                y = y == 1
                labels.append(y[:,-1])
                UpU,UpL = model(x,mask,basisR,basisI)
                score=getAnomScore1(x,UpU,UpL)
                #torch.true_divide(torch.abs(x-xu),xstd).sum(dim=(-1))
                scores.append(score)
                test_loss += loss_fn(UpU,x).item()+loss_fn(UpL,x).item()
        test_loss /= num_batches
        scores=torch.cat(scores,dim=0).cpu().detach().numpy()
        labels=torch.cat(labels,dim=0).cpu().detach().numpy()
        print(scores.shape,labels.shape)
        #pot_result = searchThreshold(scores, labels)
        pot_result= pot_eval(valscores,scores,labels)
        precision=pot_result['pot-precision']
        recall=pot_result['pot-recall']
        F1=pot_result['pot-f1']
        accuracy=torch.true_divide(pot_result['pot-TP']+pot_result['pot-TN'],pot_result['pot-TP']+pot_result['pot-TN']
                                   +pot_result['pot-FP']+pot_result['pot-FN']).item()
        print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
        print("precision:%.6f, recall:%.6f, F1 score:%.6f, accuracy:%.6f\n" % (precision, recall, F1, accuracy))
        print("average score:%f"%np.mean(scores))
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        accuracys.append(accuracy)
    return test_losses, precisions, recalls, F1s, accuracys

def loadModel(path,parameters,ks,ops):
    model = flourierConAmplified(parameters,ks,ops)
    model.load_state_dict(torch.load(path))
    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def getSeed():
    seed=int(time.time()*1000)%(2**32-1)
    return seed


def getops(ks,basisNum,l):
    pf=basisNum
    fs=[pf]
    ops=[]
    for i in range(l-1,-1,-1):
        pf=int((pf-ks[i])/2+1)
        fs.append(pf)
    for i in range(0,l):
        pf=(pf-1)*2+ks[i]
        ops.append(fs[-2-i]-pf)
        pf=fs[-i-2]
    return ops

def detectAnomaly(trainData,valDatas,valLabels,testDatas,labels,Masks,basisRs,basisIs,args,dataset):
    seed = 1282028438#getSeed()#
    setup_seed(seed)
    loadMod = args.loadMod != 0
    needTrain = args.needTrain != 0
    ks=[3,4,8,16]
    ops=getops(ks,args.basisNum,args.l)
    trainDataset=myDataset(trainData,Masks,basisRs,basisIs,args,"train")
    valDataLoaders=[]
    for valData,valLabel in zip(valDatas,valLabels):
        valDataset=myDataset([valData],Masks,basisRs,basisIs,args,"validate",[valLabel])
        valDataLoaders.append(DataLoader(valDataset, batch_size=args.batchSize))
    testDataLoaders=[]
    for testData,label in zip(testDatas,labels):
        testDataset=myDataset([testData],Masks,basisRs,basisIs,args,"test",[label])
        testDataLoaders.append(DataLoader(testDataset, batch_size=args.batchSize))
    trainDataLoader=DataLoader(trainDataset,batch_size=args.batchSize,shuffle=True)


    dirName = "flourierConv_"+str(dataset)+str(args.feaLen) + "_" + str(args.winLen)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    modelPath = dirName + "/Conv"+str(seed)+".pth"
    if not loadMod:
        model=flourierConAmplified(args,ks,ops).to(args.device)
    else:
        model=loadModel(modelPath,args,ks,ops).to(args.device)
    loss_fn=nn.MSELoss()
    epochs = 1
    bestF1 = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.1)
    trainStart=time.time()
    if needTrain:
        lastF1 = 0.
        count = 0
        torch.save(model.cpu().state_dict(), modelPath)
        model = model.to(args.device)
        print("Saved PyTorch Model State to " + modelPath)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(trainDataLoader,model,loss_fn,args,optimizer)
            test_loss, precision, recall, F1, accuracy, threshold, scores = validate(valDataLoaders,model,loss_fn,args)
            test_loss = torch.tensor(test_loss)
            if torch.isnan(test_loss).any():
                break
            #nx, nmask, nbasisR, nbasisI, nindex=skimming(trainDataLoader,model,args,threshold)
            #ntrSet=myNewTrset(nx,nmask,nbasisR,nbasisI,nindex)
            #trainDataLoader=DataLoader(ntrSet,batch_size=args.batchSize,shuffle=True)
            if lastF1 > F1:
                count += 1
            else:
                count = 0
            if count >= 2 or torch.isnan(test_loss).any():
                break
            lastF1 = F1
            if F1 > bestF1:
                bestF1 = F1
                torch.save(model.cpu().state_dict(), modelPath)
                model = model.to(args.device)
                print("Saved PyTorch Model State to " + modelPath)
    trainEnd=time.time()
    model=loadModel(modelPath,args,ks,ops).to(args.device)
    inferStart=time.time()
    test_loss,precision,recall,F1,accuracy=test(testDataLoaders,model,loss_fn,args,scores)
    inferEnd=time.time()
    trainCost=trainEnd-trainStart
    print(trainCost)
    with open(dirName + "/res"+str(seed)+".csv", "w") as f:
        for pr,re,f1,ac in zip(precision,recall,F1,accuracy):
            f.write("%f,%f,%f,%f\n" % (pr, re, f1,ac))
    with open(dirName+"/config"+str(seed)+".txt","w") as f:
        f.write(str(args))
    return (precision,recall,F1,accuracy)
