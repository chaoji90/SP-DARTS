import re, argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("DARTS first order")
#parser.add_argument('-f',type=str,help='Path to dataset')
parser.add_argument('-f',type=str,nargs='*',help='Path to dataset')
parser.add_argument('-n',type=str,default='',help='Path to dataset')
args = parser.parse_args()
#Path('/root/dir/sub/file.ext').stem
#seed=os.path.splitext(os.path.basename(args.f))[0].split('-')[1]

dfList=[]
for fil in args.f:
    dfList.append(pd.read_csv(fil))
mergeDf=pd.concat(dfList,sort=False)
mergeDf=mergeDf.drop(columns=['arch'])
mergeDf=mergeDf.melt('epoch',value_name='acc',var_name='datasets')
f=plt.figure(figsize=(6,5))
plot=sns.lineplot(data=mergeDf, x="epoch",y='acc',hue='datasets')
plt.title('Baseline Accuracy Curves of DARTS')
plt.legend(framealpha=0.4)
f.savefig('baseline.svg')
#plot=sns.lineplot(data=mergeDf, x="epoch",y='acc',hue='dataset',estimator='median',ci='sd')
#plot=sns.lineplot(data=mergeDf, x="epoch",y='acc',hue='dataset',estimator=np.median)


'''
for data in ['cifar10','cifar100','ImageNet16']:
    drawDf=mergeDf[[data+'-valid',data+'-test','epoch']]
    drawDf=drawDf.melt('epoch',value_name='acc',var_name='dataset')
    plot=sns.lineplot(data=drawDf, x="epoch",y='acc',hue='dataset')
'''

#plot=plot.get_figure()
#plot.savefig('acc'+args.n+'.svg')


