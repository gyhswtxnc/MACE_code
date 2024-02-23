import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def readData(dataPath,rootPath,type):
    existFile=os.listdir(rootPath)
    if type+".npy" in existFile:
        datas=np.load(rootPath+type+".npy",allow_pickle=True)
        return datas
    #dataPath = "ServerMachineDataset/test/"
    file_names=os.listdir(dataPath)
    file_names=sorted(file_names)
    datas=[]
    for name in file_names:
        file_name=dataPath+name
        temp = np.genfromtxt(file_name,dtype=np.float32,delimiter=',')
        print(temp.shape)
        datas.append(temp)
    datas=np.array(datas,dtype=object)
    np.save(rootPath+type+".npy",datas)
    return datas

def readCSV(dataPath,rootPath,types):
    existFile = os.listdir(rootPath)
    if types + ".npy" in existFile:
        datas = np.load(rootPath + types + ".npy", allow_pickle=True)
        return datas
    # dataPath = "ServerMachineDataset/test/"
    file_names = os.listdir(dataPath)
    file_names = sorted(file_names)
    datas = []
    for name in file_names:
        file_name = dataPath + name
        temp = pd.read_csv(file_name,header=None).to_numpy().squeeze()
        datas.append(temp)
    datas = np.array(datas, dtype=object)
    np.save(rootPath + types + ".npy", datas)
    return datas

def shuffleData(datas):
    ndatas=[]
    intervalNum=10
    indexs=np.arange(len(datas))
    for i in range(intervalNum):
        np.random.shuffle(indexs)
        for j in range(len(datas)):
            pickRow=indexs[j]
            intervalLen = int(len(datas[pickRow]) / intervalNum)
            if len(ndatas)==j:
                ndatas.append(datas[pickRow][i*intervalLen:i*intervalLen+intervalLen])
            else:
                ndatas[j]=np.append(ndatas[j],datas[pickRow][i*intervalLen:(i+1)*intervalLen])
    return ndatas


def divideTr_Val(datas,ratio=0.9):
    trdatas=[]
    valdatas=[]
    for data in datas:
        print(data.shape)
        trNum = int(ratio * len(data))
        trdatas.append(data[:trNum])
        valdatas.append(data[trNum:])
    return trdatas,valdatas



def transform(datas):
    for i in range(len(datas)):
        datas[i]=np.array(datas[i],dtype=np.float)

def nomalize(datas):
    for i in range(len(datas)):
        datas[i]/=np.max(datas[i],axis=0)+0.00001
    return datas


def getMask(counts,basisNum):#count: frequencyNum, dimensions
    counts=counts.transpose(1,0)
    mask=[]
    for count in counts:
        sortCount=sorted(count,reverse=True)
        threshold=sortCount[basisNum]
        pmask=count>threshold
        pos=0
        remainNum=basisNum-pmask.sum()
        addNum=0
        for i in range(pos,len(count)):
            if addNum==remainNum:
                break
            if count[i]==threshold and pmask[i]==False:
                pmask[i]=True
                addNum+=1
            if addNum==remainNum:
                break
        mask.append(pmask)
    mask=torch.stack(mask,dim=0)
    return mask


def getBasis(datas,winLen,basisNum):
    Masks=[]
    basisImags=[]#torch.sin(torch.fft.fftfreq(winLen))
    basisReals=[]#torch.cos(torch.fft.fftfreq(winLen))
    basisI = torch.sin(torch.fft.fftfreq(winLen))
    basisI=basisI.repeat(datas[0].shape[1],1)
    basisR = torch.cos(torch.fft.fftfreq(winLen))
    basisR=basisR.repeat(datas[0].shape[1],1)
    for data in datas:
        counts = torch.zeros(winLen, datas[0].shape[1])
        data=np.array(data,dtype=np.float)
        for i in range(0,len(data)-winLen):
            pdata=data[i:i+winLen]
            pdata=torch.tensor(pdata)
            freqs=torch.fft.fft(pdata,dim=-2)
            amplitude=torch.sqrt(torch.pow(freqs.real,2)+torch.pow(freqs.imag,2))
            counts+=amplitude
        mask=getMask(counts, basisNum)
        basisImags.append(basisI[mask].reshape(data.shape[1],-1))
        basisReals.append(basisR[mask].reshape(data.shape[1],-1))
        Masks.append(mask)
    Masks=torch.stack(Masks,dim=0)
    basisImags=torch.stack(basisImags,dim=0) #machines, fealen, winlen
    basisReals=torch.stack(basisReals,dim=0)
    return Masks, basisReals, basisImags



if __name__=="__main__":
    """labels = readData("ServerMachineDataset/test_label/", "ServerMachineDataset/", "labels")
    testData=readData("ServerMachineDataset/test/","ServerMachineDataset/","test")
    id=1
    print(testData[id].shape)
    plt.plot(labels[id])
    plt.plot(testData[id][:,3])
    plt.show()"""
    data=readData("ServerMachineDataset/train/","ServerMachineDataset/","data")
    Mask,real,image=getBasis(data,40,30)
    print(Mask.shape)
    print(real.shape)

