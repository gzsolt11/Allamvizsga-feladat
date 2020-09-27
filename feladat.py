import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics



def evaluate_identification_CV(df, num_folds=3):
    print("CV identification")
    print(df.shape)
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    
    model = RandomForestClassifier(n_estimators=100)   
    scoring = ['accuracy']
    scores = cross_val_score(model , X ,y , cv = num_folds)
    for i in range(0,num_folds):
        print('\tFold '+str(i+1)+':' + str(scores[ i ]))    
    print("accuracy : %0.4f (%0.4f)" % (scores.mean() , scores.std()))
    # sns.lineplot(data=scores)
    # plt.show()
    return [scores.mean(), scores.std()]

# df_train -dataframe for training containing class id in the last column
# df_test  -dataframe for testing containing class id in the last column
def evaluate_identification_Train_Test(df_train, df_test):
    # Train data
    array = df_train.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X_train = array[:, 0:nfeatures]
    y_train = array[:, -1]
    
    # Test data
    array = df_test.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X_test = array[:, 0:nfeatures]
    y_test = array[:, -1]
 
    model = RandomForestClassifier(n_estimators=100)   
    model.fit(X_train, y_train)
    scoring = ['accuracy']
    score_value = model.score(X_test, y_test)
    # print("accuracy : %0.4f" % (score_value))
    return score_value

def beolvas():
    fn = pathlib.Path(__file__).parent / 'sources'
    df_easy = pd.read_csv(fn / 'easy.csv')
    df_logicalstrong = pd.read_csv(fn / 'logicalstrong.csv')
    df_strong = pd.read_csv(fn / 'strong.csv')
    return [df_easy,df_logicalstrong,df_strong]

def beolvas2():
    fn = pathlib.Path(__file__).parent / 'sources/DSL-StrongPasswordData.csv'
    df_DSL = pd.read_csv(fn)
    return df_DSL

def kilenc_jellemzo_adat(df_easy,df_logicalstrong,df_strong):
    kilenc_jel = ['meanholdtime','meanpressure','meanfingerarea','meanxaccelaration','meanyaccelaration','meanzaccelaration','velocity','totaltime','totaldistance','user_id']
    return [df_easy[kilenc_jel], df_logicalstrong[kilenc_jel], df_strong[kilenc_jel]]

def cross_validation(df_easy, df_logicalstrong, df_strong):
    easy_accuracy, easy_std = evaluate_identification_CV(df_easy,10)
    logicalstrong_accuracy, logicalstrong_std = evaluate_identification_CV(df_logicalstrong,10)
    strong_accuracy, strong_std = evaluate_identification_CV(df_strong,10)
    return easy_accuracy, easy_std, logicalstrong_accuracy, logicalstrong_std, strong_accuracy, strong_std

def fel1():
    df_easy,df_logicalstrong,df_strong = beolvas()
    df_easy_kilenc, df_logicalstrong_kilenc, df_strong_kilenc = kilenc_jellemzo_adat(df_easy, df_logicalstrong, df_strong)

    easy_accuracy, easy_std, logicalstrong_accuracy, logicalstrong_std, strong_accuracy, strong_std = cross_validation(df_easy, df_logicalstrong, df_strong)
    easy_kilenc_accuracy, easy_kilenc_std, logicalstrong_kilenc_accuracy, logicalstrong_kilenc_std, strong_kilenc_accuracy, strong_kilenc_std = cross_validation(df_easy_kilenc, df_logicalstrong_kilenc, df_strong_kilenc)
    
    table = pd.DataFrame({'Jellemző':["Összes","9"],'Easy':["%0.0f (%0.4f)" % (easy_accuracy*100 , easy_std) ,"%0.0f (%0.4f)" % (easy_kilenc_accuracy*100 , easy_kilenc_std) ] ,'Logicalstrong':["%0.0f (%0.4f)" % (logicalstrong_accuracy*100 , logicalstrong_std) ,"%0.0f (%0.4f)" % (logicalstrong_kilenc_accuracy*100 , logicalstrong_kilenc_std) ] ,'Strong':["%0.0f (%0.4f)" % (strong_accuracy*100 , strong_std) ,"%0.0f (%0.4f)" % (strong_kilenc_accuracy*100 , strong_kilenc_std) ]  })
    print(table)
    x_axis = [1,2,3]
    y_axis = [easy_accuracy , logicalstrong_accuracy, strong_accuracy]
    y_axis2 = [easy_kilenc_accuracy , logicalstrong_kilenc_accuracy, strong_kilenc_accuracy]
    table2 = pd.DataFrame({"Difficulty":x_axis, "Accuracy":y_axis})
    sns.barplot(x=table2["Difficulty"],y=table2["Accuracy"])
    plt.show()

    table3 = pd.DataFrame({"Difficulty":x_axis, "Accuracy":y_axis2})
    sns.barplot(x=table3["Difficulty"],y=table3["Accuracy"])
    plt.show()




def accuracy_counting(lista):
    List = []
    for i in range(len(lista)):
        for j in range(len(lista)):
            if i != j:
                List.append("%0.0f" % (evaluate_identification_Train_Test(lista[i][lista[1].columns], lista[j][lista[1].columns])*100))
    return List

def fel2():
    df_easy,df_logicalstrong,df_strong = beolvas()
    
    het_jel = ['meanholdtime','meanpressure','meanfingerarea','meanxaccelaration','meanyaccelaration','meanzaccelaration','velocity','user_id']
    lista2 = [df_easy[het_jel], df_logicalstrong[het_jel], df_strong[het_jel]]

    List2 = accuracy_counting(lista2)
    

    tablazat = pd.DataFrame({"Train":['Easy','Easy','Logicalstrong','Logicalstrong','Strong','Strong'], "Test":['Logicalstrong','Strong','Easy','Strong','Easy','Logicalstrong'], "Accuracy_het_jel(%)":[List2[0],List2[1],List2[2],List2[3],List2[4],List2[5]]})
    print(tablazat)

def fel3():
    df_DSL = beolvas2()

    DSL_3_accuracy, DSL_3_std = evaluate_identification_CV(df_DSL,3)
    DSL_accuracy, DSL_std = evaluate_identification_CV(df_DSL,10)
    
    table = pd.DataFrame({'index':['Összes'],'DSL_3_folds':["%0.0f (%0.4f)" % (DSL_3_accuracy*100 , DSL_3_std)] ,'DSL_10_folds':["%0.0f (%0.4f)" % (DSL_accuracy*100 , DSL_std)]})
    print(table)

def main():
    # fel1()
    # fel2()
    fel3()

main()