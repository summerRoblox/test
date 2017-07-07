

```python
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from password_config import *
from pandas import read_sql
import seaborn as sns
from numpy import median
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import euclidean
```


```python
conStr = "dbname='%s' port='%s' user='%s' host='%s' password='%s'" % (get_access_database("redshift.analyst_readonly"), get_access_port("redshift.analyst_readonly"), get_access_username("redshift.analyst_readonly"), get_access_host("redshift.analyst_readonly"), get_access_password("redshift.analyst_readonly"))
conn = psycopg2.connect(conStr)
cursor = conn.cursor()
```


```python
%matplotlib inline 
```


```python
query = """\
select * from analytics_sandbox.friend
;
    """
```


```python
df = read_sql(query, con = conn)
```


```python
plt.figure(figsize=(15, 6))
plt.scatter(df['friendcount'], df['cnt'])
```




    <matplotlib.collections.PathCollection at 0x1150cfdd8>




![png](output_5_1.png)



```python
dt_trans = df[['friendcount','cnt']]
```


```python
cursor.close()
conn.close()
```


```python
K = range(1,16)
KM = [KMeans(n_clusters=k,max_iter=100).fit(dt_trans) for k in K]
centroids = [k.cluster_centers_ for k in KM]
D_k = [cdist(dt_trans, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/dt_trans.shape[0] for d in dist]
```


```python
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
```




    <matplotlib.text.Text at 0x10ce31358>




![png](output_9_1.png)



```python
kIdx = 4
```


```python
len(dist)
```




    15




```python
df1 = df[['friendcountgroup','cnt']]
f, ax = plt.subplots(figsize=(15, 6)) 
sns.barplot(x="friendcountgroup", y="cnt", data=df1, order=['1-5','6-10','11-50','50+'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x114ea9860>




![png](output_12_1.png)



```python
centroids[4] 
```




    array([[   22.70502858,    12.97882488],
           [  187.21308436,    18.56648126],
           [   98.38251622,    16.6894206 ],
           [  101.13575758,   278.04484848],
           [   98.14393939,  1063.87121212]])




```python
KM = KMeans(n_clusters=5,max_iter=300).fit(dt_trans)
centroids = KM.cluster_centers_ 
```


```python
centroids
```




    array([[   22.31807419,    12.94674532],
           [  186.97642566,    18.56512577],
           [  100.97082067,   278.41215805],
           [   97.37652415,    16.72163914],
           [   98.14393939,  1063.87121212]])


