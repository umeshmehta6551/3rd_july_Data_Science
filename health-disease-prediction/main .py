
import os
import numpy as np
import pandas as pd
import sqlite3 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tkinter import *
from tkinter import messagebox
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

list_d=['fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',  
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine']


disease=[ 'Osteoarthristis', 'Arthritis','Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Drug Reaction', 
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'AIDS', 'Diabetes ','Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Urinary tract infection',
       'Psoriasis', 'Impetigo','Peptic ulcer diseae']

list_sym=[]
for i in range(0,len(list_d)):
    list_sym.append(0)
def gui():
  w_dis2 = Label(rt, justify=LEFT, text="Diseases Predictor using symptoms ", fg="black", bg="Mintcream")
  w_dis2.config(font=("Times",32,"bold"))
  w_dis2.grid(row=2, column=0, columnspan=2, padx=100)

  S_1List_b = Label(rt, text="Symptom 1 *", fg="Black", bg="Mintcream")
  S_1List_b.config(font=("Times",18,"bold"))
  S_1List_b.grid(row=8, column=0, pady=5, sticky=W)

  S_2List_b = Label(rt, text="Symptom 2 *", fg="Black", bg="Mintcream")
  S_2List_b.config(font=("Times",18,"bold"))
  S_2List_b.grid(row=9, column=0, pady=5, sticky=W)

  S_3List_b = Label(rt, text="Symptom 3", fg="Black",bg="Mintcream")
  S_3List_b.config(font=("Times",18,"bold"))
  S_3List_b.grid(row=10, column=0, pady=5, sticky=W)

  S_4List_b = Label(rt, text="Symptom 4", fg="Black", bg="Mintcream")
  S_4List_b.config(font=("Times",18,"bold"))
  S_4List_b.grid(row=11, column=0, pady=5, sticky=W)

  S_5List_b = Label(rt, text="Symptom 5", fg="Black", bg="Mintcream")
  S_5List_b.config(font=("Times",18,"bold"))
  S_5List_b.grid(row=12, column=0, pady=5, sticky=W)

  knn_List_b = Label(rt, text="kNN", fg="black", bg="white", width = 20)
  knn_List_b.config(font=("Times",18,"bold"))
  knn_List_b.grid(row=20, column=0, pady=5, sticky=W)

  lr_List_b = Label(rt, text="Decision Tree", fg="black", bg="white", width = 20)
  lr_List_b.config(font=("Times",18,"bold"))
  lr_List_b.grid(row=22, column=0, pady=5,sticky=W)


  ranf_List_b = Label(rt, text="Naive_Bayes", fg="black", bg="white", width = 20)
  ranf_List_b.config(font=("Times",18,"bold"))
  ranf_List_b.grid(row=24, column=0, pady=10, sticky=W)



df=pd.read_csv("Training.csv")
DF= pd.read_csv('Training.csv', index_col='prognosis')

df.replace({'prognosis':{'Hyperthyroidism':32,'Hypoglycemia':33,'Arthritis':35, 'Fungal infection':0,'Allergy':1,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,'GERD':2,'Chronic cholestasis':3,'Hepatitis E':23,'Tuberculosis':25,
    'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22, 'Paralysis (brain hemorrhage)':13,'Jaundice':14,
    'Acne':37,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,
    'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Impetigo':40, '(vertigo) Paroymsal  Positional Vertigo':36,'Osteoarthristis':34, 'Urinary tract infection':38,'Alcoholic hepatitis':24,'Psoriasis':39, }},inplace=True)



X= df[list_d]
y = df[["prognosis"]]
np.ravel(y)


read_list=pd.read_csv("Testing.csv")


read_list.replace({'prognosis':{'Hyperthyroidism':32,'Hypoglycemia':33,'Arthritis':35, 'Fungal infection':0,'Allergy':1,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,'GERD':2,'Chronic cholestasis':3,'Hepatitis E':23,'Tuberculosis':25,
    'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22, 'Paralysis (brain hemorrhage)':13,'Jaundice':14,
    'Acne':37,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,
    'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Impetigo':40, '(vertigo) Paroymsal  Positional Vertigo':36,'Osteoarthristis':34, 'Urinary tract infection':38,'Alcoholic hepatitis':24,'Psoriasis':39, }},inplace=True)

def predict_disease_no(n):
    c=0;
    k=n;
    while(k>0):
        c=c+k
        k=k-1
    return c

X_test= read_list[list_d]
y_test = read_list[["prognosis"]]
np.ravel(y_test)

def mark_disease(a,n):
    s=0;
    for i in range(0,n):
        s+=i*a
    return s

def mark_symptom(b,n):
    s=0;
    for i in range(0,n):
      if(i*b<n):
        s+=i*b
      else:
        s=s+i
    return s

def KNN():
     n=4
     if ((diseasesymptm1.get()=="choose") or (diseasesymptm2.get()=="choose")):
        predict1.set(" ")
        sym=messagebox.askokcancel("System","minimun two symptom required to preidict disease")
        if sym:
            rt.mainloop()
     else:
        a=3
        from sklearn.neighbors import KNeighborsClassifier
        z11=predict_disease_no(n)
        classify=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        classify=classify.fit(X,np.ravel(y))
        z11=mark_symptom(a,n)
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=classify.predict(X_test)
        print("------------------------kNN-------------------------------")
        print("Accuracy of knn")
        print(accuracy_score(y_test, y_pred))
        z11=mark_symptom(a,n)
        print(accuracy_score(y_test, y_pred,normalize=False))
        psymptoms = [diseasesymptm1.get(),diseasesymptm2.get(),diseasesymptm3.get(),diseasesymptm4.get(),diseasesymptm5.get()]
        z11=mark_symptom(a,n)
        for k in range(0,len(list_d)):
            for z in psymptoms:
                if(z==list_d[k]):
                    list_sym[k]=1

        inputtest = [list_sym]
        z11=predict_disease_no(n)
        predict = classify.predict(inputtest)
        predicted=predict[0]

        flag='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                flag='yes'
                break

        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)
        if (flag=='yes'):
            predict3.set(" ")
            z11=predict_disease_no(n)
            predict3.set(disease[a])
        else:
            predict3.set(" ")
            z11=predict_disease_no(n)
            predict3.set("Not Found")
        
        
        import sqlite3 
        z11=predict_disease_no(n)
        c = sqlite3.connect('mysqlite.db') 
        z11=mark_disease(a,n)
        screen = c.cursor() 
        screen.execute("CREATE TABLE IF NOT EXISTS KNearestNeighbour(Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        z11=mark_symptom(a,n)
        screen.execute("INSERT INTO KNearestNeighbour(Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?)",(diseasesymptm1.get(),diseasesymptm2.get(),diseasesymptm3.get(),diseasesymptm4.get(),diseasesymptm5.get(),predict3.get()))
        c.commit()  
        screen.close() 
        c.close()


def sum_symptom(n):
    sum=0;
    for i in range(0,n):
        sum+=i
    return sum
def predict_symptom_no(k):
    c=0;
    while(k>0):
        c=c+k
        k=k-1
    return c
def Reset():
    global prev_win

    diseasesymptm1.set("choose")
    diseasesymptm2.set("choose")
    diseasesymptm3.set("choose")
    diseasesymptm4.set("choose")
    diseasesymptm5.set("choose")


    predict1.set(" ")
    predict2.set(" ")
    predict3.set(" ")
    try:
        prev_win.destroy()
        prev_win=None
    except AttributeError:
        pass



def DecisionTree():
    n=4
    if ((diseasesymptm1.get()=="choose") or (diseasesymptm2.get()=="choose")):
        predict1.set(" ")
        sym=messagebox.askokcancel("System","minimun two symptom required to preidict disease")
        if sym:
            rt.mainloop()
    else:
        from sklearn import tree
        a=7
        c_list_f3 = tree.DecisionTreeClassifier() 
        z11=predict_disease_no(n)
        c_list_f3 = c_list_f3.fit(X,y)
        z11= mark_disease(a,n)
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=c_list_f3.predict(X_test)
        print("--------------------------Decision Tree-----------------------------")

        print("Accuracy of decision tree")
        print(accuracy_score(y_test, y_pred))
        z11=mark_disease(a,n)
        print(accuracy_score(y_test, y_pred,normalize=False))
        psymptoms = [diseasesymptm1.get(),diseasesymptm2.get(),diseasesymptm3.get(),diseasesymptm4.get(),diseasesymptm5.get()]

        for k in range(0,len(list_d)):
            for z in psymptoms:
                if(z==list_d[k]):
                    list_sym[k]=1

        inputtest = [list_sym]
        predict = c_list_f3.predict(inputtest)
        predicted=predict[0]
        print("Confusion matrix")
        z11=predict_disease_no(n)
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)

        flag='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                flag='yes'
                break

    
        if (flag=='yes'):
            predict1.set(" ")
            z11=predict_disease_no(n)
            predict1.set(disease[a])
        else:
            predict1.set(" ")
            z11=predict_disease_no(n)
            predict1.set("Not Found")
 
    
        import sqlite3 
        z11=predict_symptom_no(n)
        tap = sqlite3.connect('mysqlite.db') 
        scr = tap.cursor() 
        z11=disease_factor(a,n,a,n)
        scr.execute("CREATE TABLE IF NOT EXISTS DecisionTree(Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        z11=predict_symptom_no(n)
        scr.execute("INSERT INTO DecisionTree(Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?)",(diseasesymptm1.get(),diseasesymptm2.get(),diseasesymptm3.get(),diseasesymptm4.get(),diseasesymptm5.get(),predict1.get()))
        tap.commit() 
        z11=predict_symptom_no(n) 
        scr.close() 
        tap.close()


def Exit():
    qExit=messagebox.askyesno("System","Do you want to exit the system")
    
    if qExit:
        rt.destroy()
        exit()
def review_disease(x):
    if(x<0):
     return -1;
    else:
     return x*x;
def disease_factor(a,b,c,d):
    x=0
    x=a*b+c*d
    for i in range(0,10):
        x=x+i
    return x;
def NaiveBayes():
    n=4
    if ((diseasesymptm1.get()=="choose") or (diseasesymptm2.get()=="choose")):
        predict1.set(" ")
        sym=messagebox.askokcancel("System","minimun two symptom required to preidict disease")
        if sym:
            rt.mainloop()
    else:
        a=3
        from sklearn.naive_bayes import GaussianNB
        Gaussianb = GaussianNB()
        z11=predict_symptom_no(n)
        Gaussianb=Gaussianb.fit(X,np.ravel(y))
        z11=disease_factor(a,n,a,n)
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=Gaussianb.predict(X_test)
        z11=predict_symptom_no(n)
        print("---------------------------Naive Bayes----------------------------------")
        print("Accuracy of Navie Bayes")
        print(accuracy_score(y_test, y_pred))
        z11=disease_factor(a,n,a,n)
        print(accuracy_score(y_test, y_pred,normalize=False))
        
        z11=disease_factor(a,n,a,n)
        psymptoms = [diseasesymptm1.get(),diseasesymptm2.get(),diseasesymptm3.get(),diseasesymptm4.get(),diseasesymptm5.get()]
        for k in range(0,len(list_d)):
            for z in psymptoms:
                if(z==list_d[k]):
                    list_sym[k]=1

        inputtest = [list_sym]
        z11=disease_factor(a,n,a,n)
        predict = Gaussianb.predict(inputtest)
        predicted=predict[0]
        print("Confusion matrix")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix) 
        flag='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                flag='yes'
                break
        if (flag=='yes'):
            predict2.set(" ")
            z11=predict_symptom_no(n)
            predict2.set(disease[a])
        else:
            predict2.set(" ")
            z11=predict_symptom_no(n)
            predict2.set("Not Found")


        import sqlite3 
        connection=sqlite3.connect('mysqlite.db') 
        screen =connection.cursor() 
        screen.execute("CREATE TABLE IF NOT EXISTS NaiveBayes(Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        screen.execute("INSERT INTO NaiveBayes(Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?)",(diseasesymptm1.get(),diseasesymptm2.get(),diseasesymptm3.get(),diseasesymptm4.get(),diseasesymptm5.get(),predict2.get()))
        connection.commit()  
        screen.close() 
        connection.close()

rt = Tk()
predict1=StringVar()
predict2=StringVar()
predict3=StringVar()




rt.configure(background='Mintcream')
rt.title('')
rt.resizable(0,0)

diseasesymptm1 = StringVar()
diseasesymptm1.set("choose")

diseasesymptm2 = StringVar()
diseasesymptm2.set("choose")

diseasesymptm3 = StringVar()
diseasesymptm3.set("choose")

diseasesymptm4 = StringVar()
diseasesymptm4.set("choose")

diseasesymptm5 = StringVar()
diseasesymptm5.set("choose")


prev_win=None
from tkinter import messagebox
gui()
OPTIONS = sorted(list_d)
Sym1 = OptionMenu(rt, diseasesymptm1,*OPTIONS)
Sym1.grid(row=8, column=1)

Sym2 = OptionMenu(rt, diseasesymptm2,*OPTIONS)
Sym2.grid(row=9, column=1)

Sym3 = OptionMenu(rt, diseasesymptm3,*OPTIONS)
Sym3.grid(row=10, column=1)

Sym4 = OptionMenu(rt, diseasesymptm4,*OPTIONS)
Sym4.grid(row=11, column=1)

Sym5 = OptionMenu(rt, diseasesymptm5,*OPTIONS)
Sym5.grid(row=12, column=1)

ftdst = Button(rt, text="Prediction 3", command=NaiveBayes,bg="black",fg="white")
ftdst.config(font=("Times",18,"bold"))
ftdst.grid(row=10, column=2,padx=5)

ftlr = Button(rt, text="Prediction 2", command=DecisionTree,bg="black",fg="white")
ftlr.config(font=("Times",18,"bold"))
ftlr.grid(row=8, column=2,padx=5)

ftkn = Button(rt, text="Prediction 1", command=KNN,bg="black",fg="white")
ftkn.config(font=("Times",18,"bold"))
ftkn.grid(row=6, column=2,padx=5)

ftrs = Button(rt,text="Reset Inputs", command=Reset,bg="red",fg="black",width=15)
ftrs.config(font=("Times",18,"bold"))
ftrs.grid(row=12,column=2,padx=5)

ftex = Button(rt,text="Exit System", command=Exit,bg="red",fg="black",width=15)
ftex.config(font=("Times",18,"bold"))
ftex.grid(row=14,column=2,padx=5)



event_0=Label(rt,font=("Times",18,"bold"),text="kNN",height=1,bg="white"
         ,width=40,fg="black",textvariable=predict3,relief="groove").grid(row=20, column=2, padx=10)

event_1=Label(rt,font=("Times",18,"bold"),text="Decision Tree",height=1,bg="white"
         ,width=40,fg="black",textvariable=predict1,relief="groove").grid(row=22, column=2, padx=10)

event_2=Label(rt,font=("Times",18,"bold"),text="Naive Bayes",height=1,bg="white"
         ,width=40,fg="black",textvariable=predict2,relief="groove").grid(row=24, column=2, padx=10)



rt.mainloop()
