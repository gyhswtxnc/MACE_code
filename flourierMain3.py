#main for flourier10 SMD group 2
#epoch=1
from flourierConv10 import detectAnomaly
from utils import readCSV,divideTr_Val,transform,nomalize,readData,getBasis
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description='[VAE]')
parser.add_argument('--feaLen',type=int, required=False, default=38,help='feature length')
parser.add_argument('--winLen',type=int,required=False,default=40,help='window length')
parser.add_argument('--basisNum',type=int,required=False,default=20,help='number of signal basis')
parser.add_argument('--l',type=int,required=False,default=2,help="convolutional layers")
parser.add_argument('--gamma',type=int,required=False,default=11,help="odd number")
parser.add_argument('--loadMod',type=int,required=False,default=0,help='load model from file')
parser.add_argument('--device',type=str,required=False,default='cpu',help='train on which device')
parser.add_argument('--batchSize',type=int,required=False,default=100,help='batch size')
parser.add_argument('--needTrain',type=int,required=False,default=1,help='need train or just inference')
parser.add_argument('--lr',type=float,required=False,default=0.00001,help='learning rate')
parser.add_argument('--sigma',type=int,required=False,default=5,help='scaling parameter')
parser.add_argument('--timek',type=int,required=False,default=5,help='amplify anomaly in time domain')
parser.add_argument('--tgamma',type=int,required=False,default=11,help='gamma for time domain')
parser.add_argument('--tsigma',type=int,required=False,default=5,help='sigma for time domain')

args = parser.parse_args()
datas=readData("ServerMachineDataset/train2/","ServerMachineDataset/","train2")
testData=readData("ServerMachineDataset/test2/","ServerMachineDataset/","test2")
labels=readData("ServerMachineDataset/test_label2/","ServerMachineDataset/","labels2")
"""datas=readData("JumpStartData/Dataset1/train/","JumpStartData/Dataset1/","train")
testData=readData("JumpStartData/Dataset1/test/","JumpStartData/Dataset1/","test")
labels=readData("JumpStartData/Dataset1/test_label/","JumpStartData/Dataset1/","labels")"""
"""datas=readCSV("JumpStartData/Dataset1/train/","JumpStartData/Dataset1/","train")
adaptData=readCSV("JumpStartData/Dataset1/adapt/","JumpStartData/Dataset1/","adapt")
testData=readCSV("JumpStartData/Dataset1/test/","JumpStartData/Dataset1/","test")
labels=readCSV("JumpStartData/Dataset1/test_label/","JumpStartData/Dataset1/","labels")"""
"""datas=readCSV("SMAP/train/","SMAP/","train")
adaptData=readCSV("SMAP/adapt/","SMAP/","adapt")
testData=readCSV("SMAP/test/","SMAP/","test")
labels=readCSV("SMAP/label/","SMAP/","label")"""
dataset="SMD_"
transform(datas)
transform(testData)
transform(labels)
nomalize(datas)
nomalize(testData)

"""trains=datas[[0,1,5,6]]
testD=testData[[0,1,2,4,5,6]]
testL=labels[[0,1,2,4,5,6]]
valdatas=testData[[3,7]]
valLabels=labels[[3,7]]"""

"""results=[]
for i in range(len(testData)):
    trains=datas[i:i+1]
    testD=testData[i:i+1]
    testL=labels[i:i+1]
    valdatas=testData[i:i+1]
    valLabels=labels[i:i+1]
    #valdatas,testD=divideTr_Val(testD,0.3)
    #valLabels,testL=divideTr_Val(testL,0.3)
    Masks,basisRs,basisIs=getBasis(trains,args.winLen,args.basisNum)
    datasetID=dataset+str(i)
    result=detectAnomaly(trains,valdatas,valLabels,testD,testL,Masks,basisRs,basisIs,args,datasetID)
    results.append(result)
results=np.array(results).reshape(4,-1).transpose(1,0)"""


trains=datas[[0,1,2,6,7,8]]
testD=testData[[0,1,2,6,7,8]]
testL=labels[[0,1,2,6,7,8]]
valdatas=testData[[3,4,5]]
valLabels=labels[[3,4,5]]
Masks,basisRs,basisIs=getBasis(trains,args.winLen,args.basisNum)
datasetID=dataset+str(0)
result=detectAnomaly(trains,valdatas,valLabels,testD,testL,Masks,basisRs,basisIs,args,datasetID)
results=np.array(result).reshape(4,-1).transpose(1,0)

print(results.shape)
if not os.path.exists("flourier_"+dataset):
    os.mkdir("flourier_"+dataset)
np.savetxt("flourier_"+dataset+"/Result.csv",results,fmt='%f',delimiter=',')
