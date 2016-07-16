import matplotlib.pyplot as plt
import pandas as pd
import glob

files = glob.glob('*.csv')
L1=['1.csv','2.csv','3.csv','4.csv','5.csv','all.csv']
L2=['12.csv','13.csv','14.csv','15.csv','23.csv','24.csv','25.csv','34.csv','35.csv','45.csv','all.csv']
L3=['123.csv','124.csv','125.csv','234.csv','235.csv','245.csv','345.csv','all.csv']
L4=['1234.csv','1235.csv','1245.csv','1345.csv','2345.csv','all.csv']
base = 82.24
bestAcc, bestIdx, Rank=[],[],[]

# for L in [L1,L2,L3,L4]:
#     for f in L:
#         name = f.split('.')[0]
#         T = pd.read_csv(f)
#         T['index']=T['Unnamed: 0'] # fix bug
#         above = T[T['test accuracy'] >= base]
#         plt.plot(above['index'],above['test accuracy'])
#         for idx,row in above.iterrows():
#             x,y = row['index'], row['test accuracy']
#             plt.annotate(name,xy=(x,y),xytext=(x,y))
#     plt.axhline(y=base, label='B (no transfer)', linestyle='-.', color='r')
#     plt.legend()
#     plt.show()

allAcc = None
for f in files:
    name = f.split('.')[0]
    T = pd.read_csv(f)
    T['index']=T['Unnamed: 0'] # fix bug
    #above = T[T['test accuracy'] >= base]
    #plt.plot(above['index'],above['test accuracy'])
    #for idx,row in above.iterrows():
        #x,y = row['index'], row['test accuracy']
        #plt.plot([x],[y],marker='o',markersize=5)
        #plt.annotate(name,xy=(x,y),xytext=(x,y))
    m = T.loc[T['test accuracy'].idxmax()]
    print(m)
    bestAcc.append(m['test accuracy'])
    bestIdx.append(m['index'])
    Rank.append(name)
    if name == 'all':
        allAcc = m['test accuracy']
#plt.axhline(y=base, label='B (no transfer)', linestyle='-.', color='r')
#plt.show()


df = pd.DataFrame({'test_accuracy':bestAcc,'index':bestIdx,'transferred_layers':Rank})
for idx,row in df.iterrows():
    x,y = row['index'], row['test_accuracy']
    plt.plot([x],[y],marker='o',markersize=5)
    plt.annotate(row['transferred_layers'],xy=(x,y),xytext=(x,y))
plt.axhline(y=base, label='B (no transfer)', linestyle='-.', color='r')
plt.axhline(y=allAcc, label='transfer all layers', linestyle='-.', color='b')
plt.xlabel('# Rank1 component transferred')
plt.ylabel('test accuracy')
plt.title('transferred part layers[best]')
plt.legend()
plt.savefig('all.png')