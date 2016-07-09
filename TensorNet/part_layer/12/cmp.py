import matplotlib.pyplot as plt
import pandas as pd

T = pd.read_csv('test_acc.csv').sort(['index'])
A = pd.read_csv('all.csv').sort(['index'])
df = pd.DataFrame({'transfer all layers':A['test accuracy'].values, 'transfer layer:12':T['test accuracy'].values}, index=T['index'].values)
df.plot()
plt.axhline(y=85.62, label='B (no transfer)', linestyle='-.', color='r')
plt.xlabel('# Rank1 component transferred')
plt.ylabel('test accuracy')
plt.legend()
plt.savefig('cmp_to_all.png')