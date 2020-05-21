import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

'''
판다스 DataFrame
'''

# UCI 데이터베이스에서 붓꽃 데이터셋을 가져온다
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# 데이터 유형을 살펴본다
print(df.info())
print('')

# 숫자 칼럼의 통계 정보를 조회한다
print(df.describe())
print('')

# 데이터의 첫 10개 로우를 살펴본다
print(df.head(10))
print('')

# sepal_length가 5.0보다 큰 로우만 선택한다
df2 = df.loc[df['sepal_length'] > 5.0, ]

'''
판다스를 활용한 데이터 시각화
'''

# 클래스별로 마커 모양을 지정한다
marker_shapes = ['.', '^', '*']

# 점차트를 그린다
for i, species in enumerate(df['class'].unique()):
  if i == 0:
    ax = df[df['class'] == species].plot.scatter(x='sepal_length', y='sepal_width', marker=marker_shapes[i], s=100,title="Sepal Width vs Length by Species", label=species, figsize=(10,7))
  else:
    df[df['class'] == species].plot.scatter(x='sepal_length', y='sepal_width', marker=marker_shapes[i], s=100, title="Sepal Width vs Length by Species", label=species, ax=ax)
plt.show()
plt.clf()

# 히스토그램을 그린다
df['petal_length'].plot.hist(title='Histogram of Petal Length')
plt.show()

# 상자차트를 그린다
df.plot.box(title='Boxplot of Sepal Length & Width, and Petal Length & Width')
plt.show()

'''
판다스를 활용한 데이터 전처리
'''

# 변주형 범주 인코딩
df2 = pd.DataFrame({'Day': ['Monday','Tuesday','Wednesday',
                           'Thursday','Friday','Saturday',
                           'Sunday']})
                           
# 원핫 인코딩
print(pd.get_dummies(df2))
print('')

# 결측값을 보간한다
# 붓꽃 데이터셋을 다시 가져온다
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# 로우 10개를 무작위로 고른다
random_index = np.random.choice(df.index, replace= False, size=10)

# 무작위로 고른 로우의 sepal_length 값을 None으로 바꾼다
df.loc[random_index,'sepal_length'] = None

# 결측값 위치 확인
print(df.isnull().any())
print('')

# 결측값 제거
print("Number of rows before deleting: %d" % (df.shape[0]))
df2 = df.dropna()
print("Number of rows after deleting: %d" % (df2.shape[0]))
print('')

# 결측값을 평균값으로 대체
df.sepal_length = df.sepal_length.fillna(df.sepal_length.mean())

# 결측값이 남아 있는지 확인한다
print(df.isnull().any())
print('')
