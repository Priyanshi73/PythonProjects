#!/usr/bin/env python
# coding: utf-8

# In[2]:
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import re
import calendar

from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import roc_auc_score,precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score

import xgboost as xgbs
import seaborn as sns
import matplotlib.pyplot as plt;


# In[56]:


demogr=pd.read_csv(r"C:\Users\Priyanshi\Downloads\python\demographic_details.csv")
med_hist=pd.read_csv(r"C:\Users\Priyanshi\Downloads\python\medical_history.csv")
med_train=pd.read_csv(r"C:\Users\Priyanshi\Downloads\python\train_share.csv")
med_test=pd.read_csv(r"C:\Users\Priyanshi\Downloads\python\test_share.csv")


# In[60]:


med_train.columns,med_test.columns


# In[61]:


demogr.shape,med_hist.shape,med_train.shape,med_test.shape


# In[62]:


demogr.nunique()


# In[63]:


med_hist.nunique()


# In[65]:


new_rec=pd.merge(demogr,med_hist,how="outer",on="PatientId")
new_rec


# In[67]:


#?pd.concat
#pd.concat([med_train,new_rec],axis=1)


# In[69]:


new_rec.PatientId.nunique(),med_train.PatientId.nunique()


# In[70]:


final_test=pd.merge(med_test,new_rec,how="left",on="PatientId")
final_test.shape


# In[72]:


final_test.head()


# In[73]:


final_train=pd.merge(med_train,new_rec,how="left",on="PatientId")
final_train.shape


# In[75]:


final_train.head()


# In[78]:


final_train.nunique()


# In[79]:


final_train.dtypes


# In[80]:


final_train["PatientId"].value_counts()


# In[81]:


final_train["No-show"].value_counts()


# In[82]:


# No- Present=1
# yes- Absent=0
final_train["No-show"]=np.where(final_train["No-show"]=='No',1,0)


# In[83]:


final_train["No-show"].value_counts()


# In[84]:


79360/(20130+79360) #percentage janta who show up


# In[85]:


for col in ["ScheduledDay","AppointmentDay"]:
    final_train[col]=pd.to_datetime(final_train[col],infer_datetime_format=True)
    final_test[col]=pd.to_datetime(final_test[col],infer_datetime_format=True)


# In[86]:


#final_train["ScheduledDay"].apply(lambda x:x.weekday()).value_counts()


# In[87]:


final_train["AppointmentDay"].apply(lambda x:x.weekday()).value_counts()


# In[88]:


final_train.head()


# In[89]:


final_train["AppointmentDay"]=np.where((final_train["AppointmentDay"]-final_train["ScheduledDay"]).dt.days<0,final_train["ScheduledDay"],
         final_train["AppointmentDay"])
final_test["AppointmentDay"]=np.where((final_test["AppointmentDay"]-final_test["ScheduledDay"]).dt.days<0,final_test["ScheduledDay"],
         final_test["AppointmentDay"])


# In[90]:


final_train["Waiting_Days"]=(final_train["AppointmentDay"]-final_train["ScheduledDay"]).dt.days
final_test["Waiting_Days"]=(final_test["AppointmentDay"]-final_test["ScheduledDay"]).dt.days


# In[91]:


final_train.tail()


# In[92]:


final_train["Waiting_Days"].median()


# In[94]:


#final_train.groupby("SMS_received")["No-show"].value_counts(normalize=True)
#med_hist.groupby("Hipertension")["Diabetes"].value_counts(normalize=True)
final_train.groupby("No-show")["SMS_received"].value_counts(normalize=True)


# In[95]:


final_train["No-show"].value_counts(normalize=True)


# In[96]:


#final_train["AppointmentDay"].apply(lambda x: x.weekday())
final_train["AppointmentDay"].dt.day_name()


# In[97]:


#final_train["WeekDay"]=final_train["AppointmentDay"].apply(lambda x: x.weekday())
final_train["WeekDay"]=final_train["AppointmentDay"].dt.day_name()
final_test["WeekDay"]=final_test["AppointmentDay"].dt.day_name()
# 0-mon 1-teus .. 5-sat


# In[101]:


final_test["WeekDay"].value_counts()


# In[98]:


final_train["AppointmentDay"].dt.month_name()


# In[99]:


final_train["Appoint_month"]=final_train["AppointmentDay"].dt.month_name()
final_test["Appoint_month"]=final_test["AppointmentDay"].dt.month_name()


# In[102]:


final_train["Appoint_month"].value_counts()


# In[103]:


final_train["Schedule_hr"]=final_train["ScheduledDay"].dt.hour
final_test["Schedule_hr"]=final_test["ScheduledDay"].dt.hour


# In[104]:


final_train["Appoint_hr"]=final_train["AppointmentDay"].dt.hour
final_test["Appoint_hr"]=final_test["AppointmentDay"].dt.hour


# In[105]:


for col in ["ScheduledDay","AppointmentDay"]:
    final_train.drop([col],1,inplace=True)
    final_test.drop([col],1,inplace=True)


# In[106]:


final_train.head()


# In[107]:


final_test.head()


# In[108]:


final_train["SMS_received"].value_counts()


# In[109]:


final_train["Age"].value_counts()


# In[110]:


final_train["Gender"].value_counts() #ohe


# In[111]:


#final_train.groupby(["Gender","No-show"])["No-show"].counts()
final_train.groupby("Gender")["No-show"].value_counts(normalize=True)


# In[112]:


final_train["Neighbourhood"].value_counts()


# In[113]:


final_train["Neighbourhood"]=final_train["Neighbourhood"].str.split(" ").str[0]


# In[114]:


round(final_train.groupby("Neighbourhood")["No-show"].mean(),2)


# In[115]:


def mapping_func(df,x,y,prefix="neigh_"):
    probs=round(df.groupby(x)[y].mean(),2).to_dict()
    mapping_dict=dict()
    for k,v in probs.items():
        mapping_dict[k]=prefix+str(v).replace(".","")
    return mapping_dict    


# In[116]:


final_train["Neighbourhood"]=final_train["Neighbourhood"].map(mapping_func(final_train,"Neighbourhood","No-show"))


# In[117]:


final_train["Neighbourhood"].value_counts() #ohe


# In[118]:


final_train.dtypes


# In[119]:


final_train.isnull().sum()


# In[120]:


x_train,x_test=train_test_split(final_train,test_size=.2,random_state=1)


# In[121]:


x_train1=x_train.drop(["No-show","AppointmentID"],1)
y_train1=x_train["No-show"]


# In[122]:


x_test1=x_test.drop(["No-show","AppointmentID"],1)
y_test1=x_test["No-show"]


# In[123]:


x_train.shape,x_test.shape,x_train1.shape,x_test1.shape


# In[124]:


num_col=x_train1.select_dtypes(np.number).columns


# In[125]:


cat_col=x_train1.select_dtypes(object).columns


# In[126]:


pipe_num=make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
pipe_cat=make_pipeline(SimpleImputer(strategy="constant",fill_value="missing"),
                       OneHotEncoder(handle_unknown="ignore"))


# In[127]:


ctrans=make_column_transformer((pipe_num,num_col),(pipe_cat,cat_col))


# In[128]:


ctrans.fit_transform(x_train1)


# In[129]:


ctrans.transform(final_test)


# In[ ]:





# #Logistic Regression

# In[130]:


logreg=LogisticRegression(solver="liblinear",penalty="l1",random_state=1,max_iter=800,class_weight="balanced")


# In[131]:


logreg


# In[132]:


pipe=make_pipeline(ctrans,logreg)


# In[133]:


pipe


# In[134]:


pipe.fit(x_train1,y_train1)


# In[135]:


pipe.predict(x_train1)


# In[136]:


np.unique(pipe.predict(x_train1),return_counts=True)


# In[137]:


pipe.predict(x_test1)


# In[138]:


np.unique(pipe.predict(x_test1),return_counts=True)


# In[139]:


roc_auc_score(y_train1,pipe.predict_proba(x_train1)[:,1])


# In[140]:


roc_auc_score(y_test1,pipe.predict_proba(x_test1)[:,1])


# In[141]:


np.unique(pipe.predict(final_test),return_counts=True)


# In[428]:


final_pred=pipe.predict(final_test)


# In[429]:


final_pred


# In[430]:


final_pred=np.where(final_pred==1,"No","Yes")


# In[431]:


submiss_12=pd.DataFrame(data=final_pred)


# In[432]:


submiss_12


# In[433]:


submiss_12=submiss_12.rename({0:"No-show"},axis=1)


# In[434]:


submiss_12


# In[435]:


submiss_12["No-show"].value_counts(normalize=True)


# In[ ]:





# In[436]:


submiss_12=submiss_12.to_csv("Priyanshi_Project_63.csv",index=False)


# In[ ]:





# #GridSearchCV

# In[ ]:


#Regularization


# In[142]:


params={}
params["logisticregression__penalty"]=["l1","l2"]
params["logisticregression__C"]=[.1,.01,1,10]


# In[143]:


logreg=LogisticRegression(solver="liblinear",random_state=1)
pipe=make_pipeline(ctrans,logreg)


# In[144]:


grid=GridSearchCV(pipe,params,cv=5,scoring="accuracy")


# In[145]:


grid


# In[146]:


grid.fit(x_train1,y_train1)


# In[147]:


grid.predict(x_train1)


# In[148]:


grid.predict(x_test1)


# In[149]:


np.unique(grid.predict(x_train1),return_counts=True)


# In[150]:


np.unique(grid.predict(x_test1),return_counts=True)


# In[151]:


roc_auc_score(y_train1,grid.predict_proba(x_train1)[:,1])


# In[152]:


roc_auc_score(y_test1,grid.predict_proba(x_test1)[:,1])


# In[153]:


np.unique(grid.predict(final_test),return_counts=True)


# #XgBoost

# In[149]:


parameters={
    'n_estimators':[20,40,60,80,100],
    'max_depth':range(2,10,1),
    'learning_rate':[.1,.01,.05],
    'reg_alpha':[.1,.01,1,10],
    'reg_lambda':[.1,.01,1,10]
}


# In[150]:


parameters.items()


# In[151]:


strings="xgb__"

xgb_params={}
for k,v in parameters.items():
    xgb_params[strings+k]=v


# In[152]:


xgb_params


# In[153]:


ctrans.fit_transform(x_train1)


# In[154]:


xgb=xgbs.XGBClassifier()
pipe=Pipeline([('Columntranfer',ctrans),('xgb',xgb)])


# In[155]:


grid=RandomizedSearchCV(pipe,xgb_params,cv=5,scoring="accuracy")


# In[156]:


grid


# In[157]:


grid.fit(x_train1,y_train1)


# In[158]:


grid.predict(x_train1)


# In[159]:


grid.predict(x_test1)


# In[160]:


roc_auc_score(y_train1,grid.predict_proba(x_train1)[:,1])


# In[161]:


roc_auc_score(y_test1,grid.predict_proba(x_test1)[:,1])


# In[162]:


np.unique(grid.predict(x_train1),return_counts=True)


# In[163]:


np.unique(grid.predict(x_test1),return_counts=True)


# In[164]:


np.unique(grid.predict(final_test),return_counts=True)


# In[165]:


grid.best_params_


# In[121]:


final_pred1=grid.predict(final_test)


# In[123]:


final_pred1=np.where(final_pred1==1,"No","Yes")


# In[124]:


final_pred1


# In[125]:


submiss_2=pd.DataFrame(data=final_pred1)


# In[126]:


submiss_2


# In[131]:


submiss_2=submiss_2.rename({0:"No-show"},axis=1)


# In[132]:


submiss_2=submiss_2.to_csv("Priyanshi_Project_62.csv",index=False)


# In[ ]:




