#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:28:49 2023

@author: imin-u
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 데이터 프레임 로드
df = pd.read_csv("/Users/imin-u/Desktop/school/23년 1학기/기계학습/handspan.txt",delimiter='\t')  
#문제 2_1번 결측치 여부 확인
df_check=df.isnull().sum()
df_check
#문제 2_1번 데이터 전처리 필요 여부
#필요없음

# 독립 변수와 종속 변수 분리
X = df[['Height', 'HandSpan']]
y = df['Sex']

# 문제 2_3번
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#문제 2_4

# 로지스틱 회귀 모델 학습
log = LogisticRegression()
log.fit(X_train, y_train)

y_pred = log.predict(X_test)

# 분류 결과 평가
print(classification_report(y_test, y_pred))

intercept = log.intercept_
coefficients = log.coef_

print("절편:", intercept)
print("회귀계수:", coefficients)

#문제 2_5
# 테스트 데이터에 대한 예측값 구하기
y_pred = log.predict(X_test)

print("예측값:", y_pred)

#문제 2_6
# 오차행렬 구하기
mat = confusion_matrix(y_test, y_pred)
print("오차행렬: ")
print(mat)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 정밀도 계산
precision = precision_score(y_test, y_pred, pos_label='Female')
print("정밀도:", precision)

# 재현율 계산
recall = recall_score(y_test, y_pred, pos_label='Female')
print("재현율:", recall)

# F1 스코어 계산
f1 = f1_score(y_test, y_pred, pos_label='Female')
print("F1 스코어:", f1)

# ROC 기반 AUC 스코어 계산 
y_test = np.where(y_test == 'Female', 0, 1)
y_pred = np.where(y_pred == 'Female', 0, 1)

roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC 스코어:", roc_auc)


















