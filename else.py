import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from lightgbm import Booster as lgbm_Booster
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

data_path = 'C:/Users/Administrator/Desktop/kid/'

train = pd.read_csv(data_path + 'f_train_20180204.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'f_test_a_20180204.csv', encoding='gb2312')

# X, y = merge[:n_train], train_y[:n_train]
# df = pd.concat([X,y],axis=1)
# def VAR_level(line):
#     a1 = line['ALT']
#     a2 = line['AST']
#     if a1 > 40 and a2 > 40:
#         return "ALT_40-"
#     # elif a1 >= 40 and a1 < 70:
#     #     return "ALT_30-100"
#     # elif a1 >= 70 and a1 < 100:
#     #     return "ALT_60-100"
#     # elif a1 >= 100:
#     #     return "ALT_100+"
#     else:
#         return "ALT_na"
#
# df['VAR_LEVEL'] = df.apply(VAR_level, axis=1)
# pg = df['label'].groupby(df['VAR_LEVEL'])
# print(pg.mean())
#
# def VAR_level(line):
#     a1 = line['AST']/line['ALT']
#     if a1 < 1:
#         return "sys_90-"
#     elif a1 >= 1 and a1 < 2:
#         return "sys_90-130"
#     elif a1 >= 2:
#         return "sys_130-180"
#     else:
#         return "sys_na"
#
# df['VAR_LEVEL'] = df.apply(VAR_level, axis=1)
# pg = df['label'].groupby(df['VAR_LEVEL'])
# print(pg.mean())
# #

#duck
def make_feat(train, test):
    #/对缺失值超过一半的特征进行删除/
    #/对于基因缺失采用0值填充/
    # 对于'孕次', '产次', 'DM家族史','BMI分类'顺序变量采用众数填充/
    #/对于连续型变量进行0-1标准化，认为绝对值大于3的为异常点，同缺失值,选择以中位数填充
    merge = pd.concat([train, test])
    n_train = len(train)
    train_y = merge['label']
    merge = merge.drop(['label','id'], axis=1)
    merge = merge.reset_index(drop=True)
    merge.loc['Row_sum'] = merge.isnull().apply(lambda x: x.sum())  ##计算每列有多少缺失值
    merge = merge.drop(merge.loc[:, merge.loc['Row_sum'] > 600].columns, axis=1)  #drop 缺失值超过一半的
    merge = merge.drop(['Row_sum'],axis=0)
    merge = merge.sort_index(axis=1, ascending=False)

    # 年龄
    def age_level(line):
        age = line['年龄']
        if age < 25:
            return "age_0_25"
        elif age >= 25 and age < 30:
            return "age_25_30"
        elif age >= 30 and age < 40:
            return "age_30_40"
        elif age >= 40 and age < 50:
            return "age_40_48"
        else:
            return "age_na_"

    merge['年龄_LEVEL'] = merge.apply(age_level, axis=1)
    d_age = pd.get_dummies(merge['年龄_LEVEL'], prefix="年龄")
    merge = pd.concat([d_age, merge], axis=1)
    merge = merge.drop(['年龄_LEVEL'], axis=1)

    # 收缩压
    def sys_level(line):
        a1 = line['收缩压']
        if a1 < 90:
            return "sys_90-"
        elif a1 >= 90 and a1 < 130:
            return "sys_90-130"
        elif a1 >= 130 and a1 < 180:
            return "sys_130-180"
        else:
            return "sys_na"

    merge['sys_LEVEL'] = merge.apply(sys_level, axis=1)
    d_sys = pd.get_dummies(merge['sys_LEVEL'])
    merge = pd.concat([d_sys, merge], axis=1)
    merge = merge.drop(['sys_LEVEL'], axis=1)

    # 收缩压+舒张压
    def bld_level(line):
        a1 = line['收缩压'] + line['舒张压']
        if a1 < 150:
            return "bloodT_150-"
        elif a1 >= 150 and a1 < 200:
            return "bloodT_150-200"
        elif a1 >= 200 and a1 < 260:
            return "bloodT_200-260"
        else:
            return "bloodT_na"

    merge['bld_LEVEL'] = merge.apply(bld_level, axis=1)
    d_bld = pd.get_dummies(merge['bld_LEVEL'])
    merge = pd.concat([d_bld, merge], axis=1)
    merge = merge.drop(['bld_LEVEL'], axis=1)

    # wbc
    def wbc_level(line):
        a1 = line['wbc']
        if a1 < 8:
            return "wbc_8-"
        elif a1 >= 8 and a1 < 14:
            return "wbc_8-14"
        elif a1 >= 14 and a1 < 21:
            return "wbc_14-21"
        else:
            return "wbc_na"

    merge['wbc_LEVEL'] = merge.apply(wbc_level, axis=1)
    d_wbc = pd.get_dummies(merge['wbc_LEVEL'])
    merge = pd.concat([d_wbc, merge], axis=1)
    merge = merge.drop(['wbc_LEVEL'], axis=1)

    # 'ApoA1/ApoB
    def Ap_level(line):
        a1 = line['ApoA1'] / line['ApoB']
        if a1 < 1:
            return "Ap_1-"
        elif a1 >= 1 and a1 < 3.7:
            return "Ap_1-3.7"
        elif a1 >= 3.7 and a1 < 15:
            return "Ap_3.7-15"
        else:
            return "Ap_na"

    merge['Ap_LEVEL'] = merge.apply(Ap_level, axis=1)
    d_Ap = pd.get_dummies(merge['Ap_LEVEL'])
    merge = pd.concat([d_Ap, merge], axis=1)
    merge = merge.drop(['Ap_LEVEL'], axis=1)

    # 孕前体重
    def wei_level(line):
        a1 = line['孕前体重']
        if a1 < 50:
            return "wei_50-"
        elif a1 >= 50 and a1 < 70:
            return "wei_50-70"
        elif a1 >= 70 and a1 < 80:
            return "wei_70-80"
        elif a1 >= 80 and a1 < 100:
            return "wei_80-100"
        else:
            return "wei_na"

    merge['wei_LEVEL'] = merge.apply(wei_level, axis=1)
    d_wei = pd.get_dummies(merge['wei_LEVEL'])
    merge = pd.concat([d_wei, merge], axis=1)
    merge = merge.drop(['wei_LEVEL'], axis=1)

    # TG
    def TG_level(line):
        a1 = line['TG']
        if a1 < 1.5:
            return "TG_1.5-"
        elif a1 >= 1.5 and a1 < 5.8:
            return "TG_1.5-5.8"
        elif a1 >= 5.8 and a1 < 10:
            return "TG_5.8-10"
        else:
            return "TG_na"

    merge['TG_LEVEL'] = merge.apply(TG_level, axis=1)
    d_TG = pd.get_dummies(merge['TG_LEVEL'])
    merge = pd.concat([d_TG, merge], axis=1)
    merge = merge.drop(['TG_LEVEL'], axis=1)

    # ALT
    def ALT_level(line):
        a1 = line['ALT']
        a2 = line['AST']
        if a1 > 40 and a2 > 40:
            return "ALT_+"
        else:
            return "ALT_-"

    merge['ALT_LEVEL'] = merge.apply(ALT_level, axis=1)
    alt = pd.get_dummies(merge['ALT_LEVEL'])
    merge = pd.concat([alt, merge], axis=1)
    merge = merge.drop(['ALT_LEVEL'], axis=1)

    # VAR
    def VAR_level(line):
        a1 = line['VAR00007']
        if a1 < 1.3:
            return "VAR_1.3-"
        elif a1 >= 1.3 and a1 < 1.6:
            return "VAR_1.3-1.6"
        elif a1 > 1.6:
            return "VAR_1.6+"
        else:
            return "VAR_na"

    merge['VAR_LEVEL'] = merge.apply(VAR_level, axis=1)
    var = pd.get_dummies(merge['VAR_LEVEL'])
    merge = pd.concat([var, merge], axis=1)
    merge = merge.drop(['VAR_LEVEL'], axis=1)

    # 孕次
    def pg_level(line):
        pg = line['孕次']
        if pg < 3:
            return "pg_0_3"
        else:
            return "pg_3_"

    merge['pg_LEVEL'] = merge.apply(pg_level, axis=1)
    d_pg = pd.get_dummies(merge['pg_LEVEL'])
    merge = pd.concat([d_pg, merge], axis=1)
    merge = merge.drop(['pg_LEVEL'], axis=1)
##################################################################
    for i in['年龄','身高','孕前体重','孕前BMI','收缩压','舒张压', '孕次',
             '糖筛孕周','wbc','ALT','AST','Cr','BUN','CHO','TG','HDLC','LDLC','ApoA1','ApoB','Lpa','hsCRP']:
        Seri= merge[i].apply(lambda x:(x-merge[i].mean())/merge[i].std())
        ind = Seri[abs(Seri.values) >= 3].index
        merge.ix[ind, i] = np.nan

##################################################################
    for i in merge.columns[51:101]:   ##for all snp
       merge[i] =  merge[i].fillna(0)   ##0 means NaN
       d = pd.get_dummies(merge[i])
       d.columns = d.columns.map(lambda x: i + '_' + str(int(x)))
       merge = pd.concat([merge, d], axis=1)

    for i in ['孕次', '产次', 'DM家族史','BMI分类']:#众数填充
        a = merge[i].mode()[0]
        merge[i] = merge[i].fillna(a)

    merge.fillna(merge.median(axis=0), inplace=True)
    # df = pd.DataFrame(StandardScaler().fit_transform(merge))

    # for i in['年龄','身高','孕前体重','孕前BMI','收缩压','舒张压',
    #          '糖筛孕周','wbc','ALT','AST','Cr','BUN','CHO','TG','HDLC','LDLC','ApoA1','ApoB','Lpa','hsCRP']:
    #     merge['log' + i] = np.log(merge[i])
        # merge['err' + i] = merge[i].apply(lambda x:x-merge[i].mean())
        # merge['errabs' + i] = merge[i].apply(lambda x:abs(x-merge[i].mean()))
    #StandardScaler().fit_transform(merge)
    #基因
    merge['SNP34+SNP37'] = merge['SNP34'] + merge['SNP37']

    # merge['log年龄'] = np.log(merge['年龄'])

    merge['wbc+BMI+年龄+TG+hs'] = merge['wbc'] + merge['孕前BMI'] + merge['年龄'] + merge['TG'] + merge['hsCRP']

    ##血压和BMI
    merge['bp*BMI'] = (merge['舒张压']+merge['收缩压'])*(merge['BMI分类']+1)/2
    ##a1 = line['ApoA1'] / line['ApoB']
    #merge['bp*BMI*logcr'] =  merge['bp*BMI'] * merge['logCr']
    merge['ApoA1/ApoB'] = merge['ApoA1'] / merge['ApoB']
    #merge['bp*BMI*logage'] = merge['bp*BMI'] * merge['log年龄']

    ##孕产
    merge['孕产数'] = merge['孕次'] + merge['产次']
    merge['孕产差'] = merge['孕次'] - merge['产次']

    ##血生化
    #/肝脏类
    merge['肝脏']  = (merge['AST']+merge['ALT'])/2
    merge['AST/ALT'] = merge['AST']/merge['ALT']
    #/肾脏类别
    merge['肾脏'] = (merge['BUN'] + merge['Cr'])/2
    #/心血管类别
    merge['心血管'] = (merge['HDLC'] + merge['LDLC']+merge['CHO'] + merge['TG'])/4
    #/炎症
    merge['炎症'] = (np.log(merge['wbc'])+merge['hsCRP'])/2




    X, y = merge[:n_train], train_y[:n_train]
    test_X = merge[n_train:]

    return X, y, test_X


X, y, test_X = make_feat(train, test)
#X.isnull().any()

# def feature_engin(f):   #二次多项式创造特征
#     poly = preprocessing.PolynomialFeatures(2)
#     f = poly.fit_transform(f)
#     f = pd.DataFrame(f)
#     return f
#
# X,test_X = feature_engin(X),feature_engin(test_X)


def Corr(X):     #特征相关系数排序
    cor = pd.DataFrame(X.apply(lambda x: abs(x.corr(y))),columns=['val'])
    return cor.sort_index(by='val',ascending=False)


# X,test_X = X[X.columns[indices[0:200]]],test_X[test_X.columns[indices[0:200]]]
# X,test_X = X[fea.head(100).index],test_X[fea.head(100).index]
##RF 选取特征
def RFS(n_estimators,max_depth,random_state):
    rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, verbose=0,random_state=random_state)
    rfc.fit(X, y)
    importances = rfc.feature_importances_
    return np.argsort(importances)[::-1]  # 降序排列

fea = Corr(X)
X,test_X = X[fea.head(200).index],test_X[fea.head(200).index]

indices1,indices2 = RFS(250,6,220),RFS(250,5,320)
X1,test_X1 = X[X.columns[indices1[0:200]]],test_X[test_X.columns[indices1[0:200]]]
X,test_X = X[X.columns[indices2[0:100]]],test_X[test_X.columns[indices2[0:100]]]

indices1,indices2 = RFS(250,6,220),RFS(250,6,320)
X1,test_X1 = X[X.columns[indices1[0:120]]],test_X[test_X.columns[indices1[0:120]]]

indices2 = RFS(250,6,320)
X,test_X = X[X.columns[indices2[0:100]]],test_X[test_X.columns[indices2[0:100]]]

indices2 = RFS(250,6,220)
X1,test_X1 = X[X.columns[indices2[0:120]]],test_X[test_X.columns[indices2[0:120]]]   #0.7017

indices2 = RFS(300,5,120)
X1,test_X1 = X[X.columns[indices2[0:100]]],test_X[test_X.columns[indices2[0:100]]]   #clf1 0.7098

indices2 = RFS(500,3,120)
X,test_X = X[X.columns[indices2[0:100]]],test_X[test_X.columns[indices2[0:100]]]   #0.7069

indices2 = RFS(500,3,80)
X2,test_X2 = X[X.columns[indices2[0:150]]],test_X[test_X.columns[indices2[0:150]]]   #clf2 0.7106

indices2 = RFS(250,6,220)
X3,test_X3 = X[X.columns[indices2[0:100]]],test_X[test_X.columns[indices2[0:100]]]   #ada

indices2 = RFS(250,6,220)
X3,test_X3 = X[X.columns[indices2[0:80]]],test_X[test_X.columns[indices2[0:80]]]   #clf2 0.705
xgb_params = {
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'max_depth':5,
    'eta': 0.02,
    'colsample': 0.6,
    'gamma': 1,
    'n_thread': 4,
    'silent': 1
}
clf_lgb_params1 = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.2,
    'min_data': 50,
    'min_hessian': 1,
    'verbose': 200,
    'silent': 1
}
Ada = AdaBoostClassifier(DecisionTreeClassifier
                         (max_depth=4,min_samples_split=20,min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=800,
                         learning_rate=0.8,
                         random_state=1020

)

clf = GradientBoostingClassifier(
                                 learning_rate = 0.01,
                                 max_depth=5,
                                 n_estimators = 500,
                                 random_state=10,
                                 max_features=50
                                 )

clf1 = GradientBoostingClassifier(
                                 learning_rate = 0.02,
                                 max_depth=3,
                                 n_estimators = 800,
                                 random_state=10,
                                 max_features=70
                                 )


clf2 = GradientBoostingClassifier(
                                 learning_rate = 0.03,
                                 max_depth=3,
                                 n_estimators = 600,
                                 random_state=10,
                                 max_features=80
                                 )

clf_xgb_params1 = {
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eta': 0.02,
    'min_child_weight': 4,
    'colsample': 0.8,
    'gamma': 2,
    'silent': 1
}

clf_lgb_params2 = {
    'learning_rate': 0.005,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'mse',
    'sub_feature': 0.5,
    'num_leaves': 70,
    'colsample_bytree': 0.3,
    'feature_fraction': 0.1,
    'min_data': 20,
    'verbose': 200,
}
# clf.fit(train_X,train_y)  # Training model
# X_gbd_pred = clf.predict(valid_X)
# print('第{}次得分:{}'.format(1, f1_score(valid_y,X_gbd_pred)))
#
# train_X, valid_X, train_y, valid_y = train_test_split( X, y, test_size=0.2, random_state=120)
#
# Ada_model = Ada.fit(train_X,train_y)
# X_Ada_pred = Ada.predict(valid_X)
# print('第{}次得分:{}'.format(1, f1_score(valid_y,X_Ada_pred)))

# K折交叉验证
print('开始CV 5折训练...')
t0 = time.time()
X_preds = np.zeros(X.shape[0])
# X_xgb_preds = np.zeros(X.shape[0])
# X_Ada_preds = np.zeros(X.shape[0])
X_Gbd1_preds = np.zeros(X.shape[0])
X_Gbd2_preds = np.zeros(X.shape[0])
X_lgbm_preds = np.zeros(X.shape[0])
X_lgbm2_preds = np.zeros(X.shape[0])
X_xgb_preds = np.zeros(X.shape[0])

test_preds = np.zeros((test_X.shape[0], 5))
kf = KFold(len(X), n_folds=5, shuffle=True,random_state=320)
for i, (train_index, valid_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    # train_X, train_y = X.iloc[train_index], y.iloc[train_index]
    # valid_X, valid_y = X.iloc[valid_index], y.iloc[valid_index]
    #
    train_X1, train_y1 = X1.iloc[train_index], y.iloc[train_index]
    valid_X1, valid_y1 = X1.iloc[valid_index], y.iloc[valid_index]

    train_X2, train_y2 = X2.iloc[train_index], y.iloc[train_index]
    valid_X2, valid_y2 = X2.iloc[valid_index], y.iloc[valid_index]

    # train_X3, train_y3 = X3.iloc[train_index], y.iloc[train_index]
    # valid_X3, valid_y3 = X3.iloc[valid_index], y.iloc[valid_index]

    lgb_train = lgb.Dataset(train_X1, train_y1)
    lgb_valid = lgb.Dataset(valid_X1, valid_y1)

    xgb_train = xgb.DMatrix(train_X1, train_y1)
    xgb_valid = xgb.DMatrix(valid_X1, valid_y1)




    watchlist = [(xgb_train,'train')]
    # xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=1000,
    #                           verbose_eval=200, evals=watchlist)
    # Ada_model = Ada.fit(train_X3,train_y3)
    Gbd1_model = clf1.fit(train_X1,train_y1)
    Gbd2_model = clf2.fit(train_X2, train_y2)
    clf_lgb_model1 = lgb.train(clf_lgb_params1, lgb_train,
                               valid_sets=[lgb_valid], num_boost_round=3000,
                               verbose_eval=False,
                               early_stopping_rounds=100)
    clf_lgb_model2 = lgb.train(clf_lgb_params2, lgb_train,
                               valid_sets=[lgb_valid], num_boost_round=3000,
                               verbose_eval=False,
                               early_stopping_rounds=50)
    watchlist = [(xgb_train, 'train')]
    clf_xgb_model1 = xgb.train(clf_xgb_params1, xgb_train, num_boost_round=2000,
                               verbose_eval=False, evals=watchlist, early_stopping_rounds=50)

    # X_xgb_pred = np.where(xgb_model.predict(xgb.DMatrix(valid_X)) >= 0.5, 1, 0)
    # X_Ada_pred = Ada_model.predict(valid_X3)
    X_Gbd1_pred = Gbd1_model.predict(valid_X1)
    X_Gbd2_pred = Gbd2_model.predict(valid_X2)
    lgb1_valid_pred = np.where(clf_lgb_model1.predict(valid_X1)>=0.38,1,0)
    lgb2_valid_pred = np.where(clf_lgb_model2.predict(valid_X1)>=0.38,1,0)
    xgb_valid_pred = np.where(clf_xgb_model1.predict(xgb.DMatrix(valid_X1),
                                            ntree_limit=clf_xgb_model1.best_ntree_limit + 20)>=0.38,1,0)

    # test_Ada_pred = Ada_model.predict(test_X3)
    test_Gbd1_pred = Gbd1_model.predict(test_X1)
    test_Gbd2_pred = Gbd2_model.predict(test_X2)
    test_lgbm_pred = clf_lgb_model1.predict(test_X1)
    test_lgbm2_pred = clf_lgb_model2.predict(test_X1)
    test_xbg_pred = clf_xgb_model1.predict(xgb.DMatrix(test_X1))
    # # test_xgb_pred = np.where(xgb_model.predict(xgb.DMatrix(test_X))>=0.5,1,0)
    # print('第{}次xgb得分:{}'.format(i, f1_score(valid_y,X_xgb_pred)))
    # print('第{}次ada得分:{}'.format(i, f1_score(valid_y3, X_Ada_pred)))
    print('第{}次gbd1得分:{}'.format(i, f1_score(valid_y1, X_Gbd1_pred)))
    print('第{}次gbd2得分:{}'.format(i, f1_score(valid_y2, X_Gbd2_pred)))
    print('第{}次lgbm得分:{}'.format(i, f1_score(valid_y1, lgb1_valid_pred)))
    print('第{}次lgbm2得分:{}'.format(i, f1_score(valid_y1, lgb2_valid_pred)))
    print('第{}次xgb得分:{}'.format(i, f1_score(valid_y1, xgb_valid_pred)))

    # X_xgb_preds[valid_index] += X_xgb_pred
    # X_Ada_preds[valid_index] += X_Ada_pred
    X_Gbd1_preds[valid_index] += X_Gbd1_pred
    X_Gbd2_preds[valid_index] += X_Gbd2_pred
    X_lgbm_preds[valid_index] += lgb1_valid_pred
    X_lgbm2_preds[valid_index] += lgb2_valid_pred
    X_xgb_preds[valid_index] += xgb_valid_pred
    #X_preds[valid_index] +=  lgb1_valid_pred + X_Gbd1_pred + X_Gbd2_pred + lgb2_valid_pred + xgb_valid_pred
    X_preds[valid_index] += 0.225*lgb1_valid_pred + 0.1*X_Gbd1_pred + 0.225*X_Gbd2_pred + 0.225*\
                                                    lgb2_valid_pred + 0.225*xgb_valid_pred

    test_preds[:, i] =  np.where((0.225*lgb1_valid_pred + 0.1*X_Gbd1_pred + 0.225*X_Gbd2_pred + 0.225*\
                                                    lgb2_valid_pred + 0.225*xgb_valid_pred)>=0.3,1,0)
#分类预测结果
X1_preds = X_preds
X1_preds = np.where(X_preds>=0.3,1,0)
sum(X_preds)
# print('线下xgb得分：{}'.format(f1_score(y,X_xgb_preds)))
#print('线下ada得分：{}'.format(f1_score(y, X_Ada_preds)))
print('线下gbd1得分：{}'.format(f1_score(y,X_Gbd1_preds)))
print('线下gbd2得分：{}'.format(f1_score(y,X_Gbd2_preds)))
print('线下lgbm得分：{}'.format(f1_score(y,X_lgbm_preds)))
print('线下lgbm2得分：{}'.format(f1_score(y,X_lgbm2_preds)))
print('线下xgb得分：{}'.format(f1_score(y,X_xgb_preds)))
print('线下总得分：{}'.format(f1_score(y,X1_preds)))
print('CV训练用时{}秒'.format(time.time() - t0))

A = pd.DataFrame({'pred': X1_preds})
A.to_csv(data_path + r'train{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')

submission =  pd.DataFrame({'pred': np.where(test_preds.sum(axis=1)>=3,1,0)})
submission =  pd.DataFrame({'pred': np.where((test_preds[:,0]*0.2+test_preds[:,1]*0.3+
                                            test_preds[:, 2]*0.2+test_preds[:,3]*0.2+test_preds[:,4]*0.1)>=0.5,1,0)})

submission.to_csv(data_path + r'new{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')


sum(submission.pred)
