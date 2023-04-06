# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:54:35 2022

@author: Admin
"""

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('housing_data_train.csv')
df.shape
#The dataset has 30,471 rows and 292 columns.
# select numerical columns - lựa chọn ra các cột có kiểu dữ liệu số
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
# select non-numeric columns - lựa chọn ra các cột có kiểu dữ liệu không phải dạng số
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
# % of values missing in each column
values_list = list()
cols_list = list()
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())*100#số giá trị trong cột là null*100 (số phần trăm giá trị trong cột có giá trị là null)
    cols_list.append(col)
    values_list.append(pct_missing)
pct_missing_df = pd.DataFrame()
pct_missing_df['col'] = cols_list
pct_missing_df['pct_missing'] = values_list
pct_missing_df.loc[pct_missing_df.pct_missing > 0].plot(kind='bar', figsize=(12,8))

plt.show()
#sau bước này hiện ra được biểu đồ đánh giá tỷ lệ missing values của mỗi cột
#tiếp theo ta sẽ loại bỏ các dữ liệu có phần trăm missing values <0.5
less_missing_values_cols_list = list(pct_missing_df.loc[(pct_missing_df.pct_missing < 0.5) & (pct_missing_df.pct_missing > 0), 'col'].values)
#print(less_missing_values_cols_list)
df.dropna(subset=less_missing_values_cols_list, inplace=True)
# dropping columns with more than 40% null values - xoá bỏ các cột có phần trăm missing values lớn hơn 40
_40_pct_missing_cols_list = list(pct_missing_df.loc[pct_missing_df.pct_missing > 40, 'col'].values)
df.drop(columns=_40_pct_missing_cols_list, inplace=True)
#The number of features in our dataset is now 286.
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
for col in numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:  # impute values only for columns that have missing values
        med = df[col].median() #impute with the median 
        #gán các giá trị còn thiếu trong mỗi cột bằng giá trị trung bình của cột đó
        #tức là với các features có phần trăm trong khoảng 0.5-40
        df[col] = df[col].fillna(med)
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
for col in non_numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:  # impute values only for columns that have missing values
        mod = df[col].describe()['top'] # impute with the most frequently occuring value - giá trị xuất hiện thường xuyên nhất
        df[col] = df[col].fillna(mod)#thay thế các dữ liệu missing bằng giá trị thường xuyên đó
print(df.isnull().sum().sum())#kiểm tra còn missing values không
#life_sq để xem xét (các) điểm ngoại lệ (giá trị bất thường)
#print(df.life_sq.describe())

df.life_sq.plot(kind='box', figsize=(12, 8))

# removing the outlier value in life_sq column
df = df.loc[df.life_sq < 7478]
# dropping duplicates by considering all columns other than ID
cols_other_than_id = list(df.columns)[1:]
df.drop_duplicates(subset=cols_other_than_id, inplace=True)
#print(df.shape)
#The number of observations is 30,434 now.
plt.show()
print("before:")
print(df.timestamp.dtype) # hiện tại kiểu dữ liệu đang là object
# converting timestamp to datetime format
df['timestamp'] = pd.to_datetime(df.timestamp, format='%Y-%m-%d')#chuyển sang dạng DateTime
print("after:")
print(df.timestamp.dtype)