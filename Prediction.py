import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from numpy import interp
import xlrd 
from pandas import DataFrame
from sklearn import linear_model 
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score,\
                                roc_curve, roc_auc_score, precision_recall_curve,auc
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder as le
from xgboost import XGBClassifier
import scikitplot as skplt
import category_encoders as ce
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from scipy import interpolate
from imblearn.under_sampling import NearMiss 
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest 
import random
import scipy
import scipy.special
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from patsy import dmatrices
import statsmodels.api as sm 
from sklearn.preprocessing import PolynomialFeatures
import warnings
import collections
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

random.seed(1)

def plot_metrics(model,target,labels, predictions):
    accuracy = accuracy_score(labels,predictions)
    precision = precision_score(labels,predictions,labels=[0],average='micro')
    recall = recall_score(labels,predictions,labels=[0],average='micro')
    f1_sco = f1_score(labels,predictions,labels=[0],average='micro')
    cm = confusion_matrix(labels, predictions)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    return accuracy,precision,recall,f1_sco,cm,sensitivity,specificity



ConfusionMatrix = collections.namedtuple('conf', ['tp','fp','tn','fn']) 

def FPR(conf_mtrx):
    return conf_mtrx.fp / (conf_mtrx.fp + conf_mtrx.tn) if (conf_mtrx.fp + conf_mtrx.tn)!=0 else 0
def TPR(conf_mtrx):
    return conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn)!=0 else 0
def calc_ConfusionMatrix(actuals, scores, threshold=0.5, positive_label=1):
    tp=fp=tn=fn=0
    bool_actuals = [act==positive_label for act in actuals]
    for truth, score in zip(bool_actuals, scores):
        if score > threshold:                      
            if truth:                              
                tp += 1
            else:                                            
                fp += 1          
        else:                                      
            if not truth:                        
                tn += 1                          
            else:                                  
                fn += 1
    return ConfusionMatrix(tp, fp, tn, fn)

def apply(actuals, scores, **fxns): 
    low = min(scores)
    high = max(scores)
    step = (abs(low) + abs(high)) / 1000
    thresholds = np.arange(low-step, high+step, step)
    confusionMatrices = []
    for threshold in thresholds:
        confusionMatrices.append(calc_ConfusionMatrix(actuals, scores, threshold))
    results = {fname:list(map(fxn, confusionMatrices)) for fname, fxn in fxns.items()}
    results["THR"] = thresholds
    return results

def ROC(actuals, scores):
    return apply(actuals, scores, FPR=FPR, TPR=TPR)


def AUC(y,y_pred):
    s = 0
    for i in np.where(y == 1):
        for j in np.where(y == 0):
            if y_pred[i] > y_pred[j]:
                s = s + 1
            elif y_pred[i] == y_pred[j]:
                s = s + 0.5 
    s = s / (np.sum(y[y == 1]) * np.sum(y[y == 0]))
    return s


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      sensitivity,
      specificity,
      keras.metrics.AUC(name='auc'),
]


EPOCHS = 20
BATCH_SIZE = 2

def make_baseline_model(input_dim,metrics = METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
      keras.layers.Dense(16, activation='relu',input_shape=(input_dim,)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias),
    ])
    model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[metrics])
    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='auc', 
            verbose=0,
            patience=10,
            mode='max',
            restore_best_weights=True)


df = pd.read_excel("../Data/CV database (2).xlsx")
df = df[df.Gender != "U"]
df.drop(["LAB","LabValue"],axis=1,inplace=True)
df.dropna(inplace=True)
le = preprocessing.LabelEncoder()
df.Gender = le.fit_transform(df.Gender)
df.age = le.fit_transform(df.age)
df['Disease'] = np.where(((df.Depression == 0) & (df.Anxiety == 0) \
                                 & (df.Schizophrenia == 0)), 0, 1)

#No Variable Selection and No Undersampling
plt.rcParams['figure.figsize'] = (15, 15)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
def noVSnoUS(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]
    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
            
            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))
            result_lst.append([t,m["label"],np.mean(accuracys),np.mean(precisions),np.mean(recalls),np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])
        
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)#,ncol=2,handlelength=4)
    plt.tight_layout()
    plt.savefig("../Final Results/noVSnoUS.eps",format='eps',dpi=300)
    plt.show()   
        
    noVSnoUS_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    noVSnoUS_result.to_csv('../Final Results/noVSnoUS_result.csv',index=False,float_format='%.4f')
    return noVSnoUS_result

noVSnoUS(df)


##########################################################################################################################################################################
#No Variable selection and with Undersampling
def noVSUS(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]
    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
        
            nr = NearMiss() 
            X_train_miss, y_train_miss = nr.fit_sample(X, y) 
            
            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X_train_miss, y_train_miss):
                X_train, X_test = X_train_miss.iloc[train_index], X_train_miss.iloc[test_index]
                y_train, y_test = y_train_miss.iloc[train_index], y_train_miss.iloc[test_index]
                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
                        
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            # Now, plot the computed values
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))

       
            result_lst.append([t,m["label"],np.mean(accuracys),np.mean(precisions),np.mean(recalls),np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])

        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
        
    plt.tight_layout()
    plt.savefig("../Final Results/noVSUS.eps",format='eps',dpi=300)
    plt.show()   

    noVSUS_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    noVSUS_result.to_csv('../Final Results/noVSUS_result.csv',index=False,float_format='%.4f')
    return noVSUS_result

noVSUS(df)


##########################################################################################################################################################################
#Variable Selection using Logit and No Undersampling
def VS_LR_noUS(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]

    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
            logit_mod = sm.Logit(y,X).fit()
            cols = logit_mod.pvalues[logit_mod.pvalues < 0.05].index

            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X[cols], y):
                X_train, X_test = X[cols].iloc[train_index], X[cols].iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
                        
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            # Now, plot the computed values
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))           
            result_lst.append([t,m["label"],cols,len(cols),np.mean(accuracys),np.mean(precisions),\                                                                                                              np.mean(recalls),np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])
        # Custom settings for the plot 
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
        
    plt.tight_layout()
    plt.savefig("../Final Results/VS_LR_noUS.eps",format='eps',dpi=300)
    plt.show()   
    
    VSnoUS_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','FeatureSet','FeatureLen','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    VSnoUS_result.to_csv('../Final Results/VS_LR_noUS.csv',index=False,float_format='%.4f')
    return VSnoUS_result

VS_LR_noUS(df)


##########################################################################################################################################################################
#Variable selection using adjr2 with no undersampling
def VS_adjr2_noUS(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]

    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
            if(t == "Depression"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','Atherosclerosis','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','BMI','E_Mycin','Clarithromycin','Z_Pak','Folate','VitB6','CoQ','Omega3FishOil']
            elif(t=="Anxiety"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','Atherosclerosis','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','BMI','E_Mycin','Clarithromycin','Z_Pak','Folate','VitB6','CoQ','Omega3FishOil']
            elif(t=="Schizophrenia"):
                cols = ['age','Hypertension','CancerMalignant','HeartFailure','CerebrovascularDisease','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','ElevatedESR','LongTermUseOfAntibiotics','Z_Pak','Folate','CoQ']
            elif(t=="Disease"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','Atherosclerosis','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','BMI','E_Mycin','Clarithromycin','Z_Pak','Folate','VitB6','CoQ','Omega3FishOil']


            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X[cols], y):
                X_train, X_test = X[cols].iloc[train_index], X[cols].iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
                        
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))

        
            result_lst.append([t,m["label"],cols,len(cols),np.mean(accuracys),np.mean(precisions),np.mean(recalls),\
                               np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])
        # Custom settings for the plot 
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
        
    plt.tight_layout()
    plt.savefig("../Final Results/VS_adjr2_noUS.eps",format='eps',dpi=300)
    plt.show()   
    
    VSnoUS_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','FeatureSet','FeatureLen','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    VSnoUS_result.to_csv('../Final Results/VS_adjr2_noUS.csv',index=False,float_format='%.4f')
    return VSnoUS_result

VS_adjr2_noUS(df)

##########################################################################################################################################################################
#Variable selection using BIC with no undersampling
def VS_bic_noUS(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]

    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
            if(t == "Depression"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','Z_Pak']
            elif(t=="Anxiety"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','Clarithromycin','Z_Pak','CoQ']
            elif(t=="Schizophrenia"):
                cols = ['Hypertension']
            elif(t=="Disease"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','Clarithromycin','Z_Pak','CoQ']


            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X[cols], y):
                X_train, X_test = X[cols].iloc[train_index], X[cols].iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
                        
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))

        
            result_lst.append([t,m["label"],cols,len(cols),np.mean(accuracys),np.mean(precisions),np.mean(recalls),\
                               np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])
        # Custom settings for the plot 
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
        
    plt.tight_layout()
    plt.savefig("../Final Results/VS_bic_noUS.eps",format='eps',dpi=300)
    plt.show()   
    
    VSnoUS_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','FeatureSet','FeatureLen','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    VSnoUS_result.to_csv('../Final Results/VS_bic_noUS.csv',index=False,float_format='%.4f')
    return VSnoUS_result

VS_bic_noUS(df)


##########################################################################################################################################################################
#Variable selection using Logit with undersampling
def VSUS_LR(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),}, 
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]
    
    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
            logit_mod = sm.Logit(y,X).fit()
            cols = logit_mod.pvalues[logit_mod.pvalues < 0.05].index


            nr = NearMiss() 
            X_train_miss, y_train_miss = nr.fit_sample(X[cols], y) 

            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X_train_miss, y_train_miss):
                X_train, X_test = X_train_miss.iloc[train_index], X_train_miss.iloc[test_index]
                y_train, y_test = y_train_miss.iloc[train_index], y_train_miss.iloc[test_index]

                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
                        
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            # Now, plot the computed values
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))
            result_lst.append([t,m["label"],cols,len(cols),np.mean(accuracys),np.mean(precisions),np.mean(recalls),\
                               np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
    plt.tight_layout()
    plt.savefig("../Final Results/VSUS_LR.eps",format='eps',dpi=300)
    plt.show()   
    VSUS_LR_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','FeatureSet','FeatureLen','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    VSUS_LR_result.to_csv('../Final Results/VSUS_LR.csv',index=False,float_format='%.4f')
    return VSUS_LR_result

VSUS_LR(df)

##########################################################################################################################################################################
#Variable selection using adjr2 with undersampling
def VSUS_adjr2(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]
    
    for ix, t in enumerate(targets):
        print(t)
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []

            if(t == "Depression"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','Atherosclerosis','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','BMI','E_Mycin','Clarithromycin','Z_Pak','Folate','VitB6','CoQ','Omega3FishOil']
            elif(t=="Anxiety"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','Atherosclerosis','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','BMI','E_Mycin','Clarithromycin','Z_Pak','Folate','VitB6','CoQ','Omega3FishOil']
            elif(t=="Schizophrenia"):
                cols = ['age','Hypertension','CancerMalignant','HeartFailure','CerebrovascularDisease','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','ElevatedESR','LongTermUseOfAntibiotics','Z_Pak','Folate','CoQ']
            elif(t=="Disease"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','Atherosclerosis','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','BMI','E_Mycin','Clarithromycin','Z_Pak','Folate','VitB6','CoQ','Omega3FishOil']



            nr = NearMiss() 
            X_train_miss, y_train_miss = nr.fit_sample(X[cols], y) 

            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X_train_miss, y_train_miss):
                X_train, X_test = X_train_miss.iloc[train_index], X_train_miss.iloc[test_index]
                y_train, y_test = y_train_miss.iloc[train_index], y_train_miss.iloc[test_index]

                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
                        
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))
            result_lst.append([t,m["label"],cols,len(cols),np.mean(accuracys),np.mean(precisions),np.mean(recalls),\
                               np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])
        # Custom settings for the plot 
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
        
    plt.tight_layout()
    plt.savefig("../Final Results/VSUS_adjr2.eps",format='eps',dpi=300)
    plt.show()   
    adj_r2_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','FeatureSet','FeatureLen','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    adj_r2_result.to_csv("../Final Results/VSUS_adjr2.csv",index=False,float_format='%.4f')
    return adj_r2_result

VSUS_adjr2(df)


##########################################################################################################################################################################
#Variable selection using BIC with undersampling
def VSUS_bic(df):
    class_var = df[['Depression', 'Anxiety','Schizophrenia','Disease']]
    targets = ['Depression', 'Anxiety','Schizophrenia','Disease']
    X = df.drop(targets,axis=1)
    result_lst = []
    models = [{'label': 'RF','model': RandomForestClassifier(),},
            {'label': 'DT','model': DecisionTreeClassifier(criterion="entropy"),},
            {'label': 'NB','model': GaussianNB(),},
            {'label': 'XGB','model': XGBClassifier(),},
            {'label':'LightGBM','model':LGBMClassifier(),},
            {'label':'DL',}]
    
    for ix, t in enumerate(targets):
        plt.subplot(2,2,ix+1)
        y = class_var[t]
        for m in models:
            tprs = []
            fprs=[]
            aucs = []
            accuracys = []
            precisions = []
            recalls = []
            f1_scores = []
            cms = []
            specificitys = []
            if(t == "Depression"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','Z_Pak']
            elif(t=="Anxiety"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','Clarithromycin','Z_Pak','CoQ']
            elif(t=="Schizophrenia"):
                cols = ['Hypertension']
            elif(t=="Disease"):
                cols = ['age','InsulinDependentDiabetes','Hypertension','Osteoarthritis','CancerMalignant','Obesity','CongenitalDiseaseOfHeart','HeartFailure','CoronaryArteryDisease','NutritionDeficiency','ElevatedCRP','LongTermUseOfAntibiotics','Clarithromycin','Z_Pak','CoQ']

            nr = NearMiss() 
            X_train_miss, y_train_miss = nr.fit_sample(X[cols], y) 

            skf = KFold(n_splits=5, shuffle=True, random_state=3)
            for train_index, test_index in skf.split(X_train_miss, y_train_miss):
                X_train, X_test = X_train_miss.iloc[train_index], X_train_miss.iloc[test_index]
                y_train, y_test = y_train_miss.iloc[train_index], y_train_miss.iloc[test_index]

                if(m["label"] != "DL"):
                    model = m['model'] 
                    model.fit(X_train, y_train) 
                    y_pred=model.predict(X_test) 
                    y_prob = model.predict_proba(X_test)[:,1].ravel()
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                else:
                    baseline_model = make_baseline_model(input_dim=X_train.shape[1])
                    baseline_history = baseline_model.fit(X_train, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks = [early_stopping],verbose=0)
                    y_pred = baseline_model.predict_classes(X_test, batch_size=BATCH_SIZE)
                    y_prob = baseline_model.predict(X_test, batch_size=BATCH_SIZE)
                    acc,prec,re,f1,cm,sensiti,specifi = plot_metrics(m,t,y_test,y_pred)
                    accuracys.append(acc)
                    precisions.append(prec)
                    recalls.append(re)
                    f1_scores.append(f1)
                    cms.append(cm)
                    specificitys.append(specifi)
                    results = ROC(y_test,y_prob)
                    fpr = results["FPR"]
                    tpr = results["TPR"]
                    fprs.append(fpr)
                    tprs.append(tpr)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    
            if(isinstance(model,RandomForestClassifier) or isinstance(model,DecisionTreeClassifier) or isinstance(model,GaussianNB) or isinstance(model,XGBClassifier) or isinstance(model,LGBMClassifier)):
                min_fpr = min([len(fprs[jx]) for jx in range(len(fprs))])
                min_tpr = min([len(tprs[jx]) for jx in range(len(tprs))])
                #min_val = min(min_fpr,min_tpr)
                for ix in range(len(fprs)):
                    if(min_fpr < len(fprs[ix])):
                        fprs[ix] = fprs[ix][: - (len(fprs[ix]) - min_fpr)] 
                        fprs[ix] = fprs[ix][:-1]
                        fprs[ix].append(0.0)
                for ix in range(len(tprs)):
                    if(min_tpr < len(tprs[ix])):
                        tprs[ix] = tprs[ix][: - (len(tprs[ix]) - min_tpr)] 
                        tprs[ix] = tprs[ix][:-1]
                        tprs[ix].append(0.0)
            mean_fpr = np.mean(fprs,axis = 0)
            mean_tpr = np.mean(tprs, axis=0)
            
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, label='%s (area = %0.4f)' % (m['label'], mean_auc))
            result_lst.append([t,m["label"],cols,len(cols),np.mean(accuracys),np.mean(precisions),np.mean(recalls),\
                               np.mean(f1_scores),np.mean(specificitys),mean_fpr,mean_tpr,mean_auc,std_auc])

        plt.plot([0, 1], [0, 1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(t)
        plt.legend(frameon=True,loc="lower right",framealpha=None,fontsize=16)
        
    plt.tight_layout()
    plt.savefig("../Final Results/VSUS_bic.eps",format='eps',dpi=300)
    plt.show()   
    bic_result = pd.DataFrame(result_lst,columns=['Disease','ModelName','FeatureSet','FeatureLen','Accuracy','Precision','Recall-Sensitivity','f1','Specificity','FPR','TPR','Mean AUC','std AUC'])
    bic_result.to_csv("../Final Results/VSUS_bic.csv",index=False,float_format='%.4f')
    return bic_result


VSUS_bic(df)