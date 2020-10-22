import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xlrd 
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder as le


df = pd.read_excel("Data/CV database (2).xlsx")
df = df[df.Gender != "U"]
df.drop(["LAB","LabValue"],axis=1,inplace=True)
df.dropna(inplace=True)
le = preprocessing.LabelEncoder()
df.Gender = le.fit_transform(df.Gender)
df.age = le.fit_transform(df.age)
df['Disease'] = np.where(((df.Depression == 0) & (df.Anxiety == 0) \
                                 & (df.Schizophrenia == 0)), 0, 1)

def odds_ratio_catAgeBMI(df, label):
    class_var = df[['Depression','Anxiety','Schizophrenia','Disease']]
    y = class_var[label]
 
    if(label == "Depression"):
        X = df.drop(['Anxiety','Schizophrenia','Disease', 'BMI'],axis=1)
        fm = "Depression ~ C(Gender, Treatment('M')) + C(age) + InsulinDependentDiabetes + Hypertension + Osteoarthritis \
                + CancerMalignant + Obesity + CongenitalDiseaseOfHeart + HeartFailure + CerebrovascularDisease + \
                Atherosclerosis + CoronaryArteryDisease + NutritionDeficiency + \
                ElevatedCRP + ElevatedESR + LongTermUseOfAntibiotics + C(BMIGroup) + \
                E_Mycin + Clarithromycin + Z_Pak + Folate + VitB6 + CoQ + \
                Omega3FishOil"
    elif(label == "Anxiety"):
        X = df.drop(['Depression','Schizophrenia','Disease', 'BMI'],axis=1)
        fm = "Anxiety ~ C(Gender, Treatment('M')) + C(age) + InsulinDependentDiabetes + Hypertension + Osteoarthritis \
            + CancerMalignant + Obesity + CongenitalDiseaseOfHeart + HeartFailure + CerebrovascularDisease + \
            Atherosclerosis + CoronaryArteryDisease + NutritionDeficiency + \
            ElevatedCRP + ElevatedESR + LongTermUseOfAntibiotics + C(BMIGroup) + \
            E_Mycin + Clarithromycin + Z_Pak + Folate + VitB6 + CoQ + \
            Omega3FishOil"
        #removed atherosclerosis, cancermalignant and congenitaldiseaseofheart
    elif(label == "Schizophrenia"):
        X = df.drop(['Depression','Anxiety','Disease', 'BMI'],axis=1)
        fm = "Schizophrenia ~ C(Gender, Treatment('M')) + C(age) + InsulinDependentDiabetes + Hypertension + Osteoarthritis \
             + Obesity  + HeartFailure + CerebrovascularDisease + \
              CoronaryArteryDisease + NutritionDeficiency + \
            ElevatedCRP + ElevatedESR + LongTermUseOfAntibiotics + C(BMIGroup) + \
            E_Mycin + Clarithromycin + Z_Pak + Folate + VitB6 + CoQ + \
            Omega3FishOil"
    elif(label == "Disease"):
        X = df.drop(['Depression','Anxiety', 'Schizophrenia', 'BMI'],axis=1)
        fm = "Disease ~ C(Gender, Treatment('M')) + C(age) + InsulinDependentDiabetes + Hypertension + Osteoarthritis \
            + CancerMalignant + Obesity + CongenitalDiseaseOfHeart + HeartFailure + CerebrovascularDisease + \
            Atherosclerosis + CoronaryArteryDisease + NutritionDeficiency + \
            ElevatedCRP + ElevatedESR + LongTermUseOfAntibiotics + C(BMIGroup) + \
            E_Mycin + Clarithromycin + Z_Pak + Folate + VitB6 + CoQ + \
            Omega3FishOil"
    
    
    logit_mod = smf.logit(formula = fm, data=X).fit()
    
    
    cols = logit_mod.pvalues[logit_mod.pvalues < 0.05].index
    p_value = logit_mod.pvalues.loc[cols]
    
    params = logit_mod.params
    conf = logit_mod.conf_int()
    conf["OR"] = np.round(params,4)
    
    conf.columns = ["Lower CI", "Upper CI", "OR"]
    conf = np.round(np.exp(conf.loc[cols,:]),4)
    conf["P-Values"] = np.round(p_value,4)
    conf["n_obs"] = conf.rename(index=num_df[label]).index
    print("odds ratio with confidence intervals")
    return(conf.sort_values(ascending=False,by='OR'))
    
    
    
    
#Odds ratio for depression
odds_ratio_catAgeBMI(df,"Depression")

#Odds ratio for anxiety
odds_ratio_catAgeBMI(df,"Anxiety")

#Odds ratio for schizophrenia
odds_ratio_catAgeBMI(df,"Schizophrenia")

#Odds ratio for depression
odds_ratio_catAgeBMI(df,"Disease")