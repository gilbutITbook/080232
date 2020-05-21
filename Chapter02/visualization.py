import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('diabetes.csv')

# 상위 5개 데이터를 출력한다
print(df.head())


# 히스토그램을 살펴본다
df.hist()
plt.tight_layout()
plt.show()


# 밀도 차트를 그린다
# 3 x 3 크기로 서브 차트를 만든다
plt.subplots(3,3,figsize=(20,20))

# 각 특징 변수의 밀도 차트를 그린다
for idx, col in enumerate(df.columns[:-1]):
    ax = plt.subplot(3,3,idx+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel= False, kde_kws={'linestyle':'-', 'color':'black', 'label':"No Diabetes"})
    sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel= False, kde_kws={'linestyle':'--', 'color':'black', 'label':"Diabetes"})
    ax.set_title(col)

# 차트가 8개 뿐이므로 마지막 9번째(우측하단) 서브 차트는 숨긴다
plt.subplot(3,3,9).set_visible(False)
plt.tight_layout()
plt.show()
