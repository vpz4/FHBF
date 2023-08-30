# -*- coding: utf-8 -*-
"""

@author: bpez
"""
# -*- coding: utf-8 -*-
"""

@author: bpez
"""
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb
import random
import sklearn.metrics as skl_metrics
import shap
import seaborn as sns
import copy
import scipy.stats as sp
import time
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import metrics
from scipy import interp
from sklearn import linear_model, naive_bayes
from scipy.stats import ranksums
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

def gradient(predt, dtrain):
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)


def hessian(predt, dtrain):
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) / np.power(predt + 1, 2))


def squared_log(predt, dtrain):
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def huber_approx_obj(predt, dtrain, h):
    d = predt - dtrain.get_label()
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def logcosh(predt, dtrain):
    x = predt - dtrain.get_label()
    grad = np.tanh(x)
    hess = 1 / np.cosh(x)**2
    return grad, hess


def modified_huber_loss(predt, y_true):
    z = predt * y_true
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.] = 0
    return loss


def Huber(yHat, y, delta):
    return np.where(np.abs(y-yHat) < delta, .5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))


def my_obj(predt, dtrain):
    y = dtrain.get_label();
    
    d_DF = pd.read_excel('results/v3/MALT/delta_temp.xlsx');
    d = d_DF['delta'].values[0];
    
    #logcosh
    f = np.log(np.cosh(predt-y));
    
    # g = modified_huber_loss(predt, y); #Huber loss
    g = Huber(predt, y, d); #Huber loss
    
    #first and second order derivative of logcosh
    [f_, f__] = logcosh(predt, dtrain);
    
    #first and second order derivative of modified huber loss
    [g_, g__] = huber_approx_obj(predt, dtrain, d); 
    # [g_, g__] = squared_log(predt, dtrain);
    
    #proposed gradient and hessian
    grad = (f_ *g) + (f * g_);
    hess = (f__ * g) + (2*f_*g_) +  (f*g__);
    
    # plt.figure(100);
    # plt.plot(f);
    # plt.plot(g);
    # plt.savefig('costs.png', 
    #             dpi=300,
    #             bbox_inches="tight");
    # plt.close();
    
    return grad, hess


def loss_dist(predt, dtrain):
    y = dtrain.get_label();
    d = 1;
    f = np.log(np.cosh(predt-y));
    g = Huber(predt, y, d);
    
    return f, g


def compute_eval_measures_dist(dlr, T, y_T, y_score, y_pred):
    [fpr, tpr, thresholds] = skl_metrics.roc_curve(y_T, y_pred);
    roc_auc = skl_metrics.auc(fpr, tpr);
    acc = skl_metrics.accuracy_score(y_T, y_score);
    recall = skl_metrics.recall_score(y_T, y_score, average = 'macro');
    prec = skl_metrics.precision_score(y_T, y_score);
    roc_auc = skl_metrics.auc(fpr, tpr);
    [TN, FP, FN, TP] = skl_metrics.confusion_matrix(y_T, y_score).ravel();
    sens = TP/(TP+FN);
    spec = TN/(TN+FP);
    
    return [acc, sens, spec, recall, prec, roc_auc, tpr, fpr];


def FHBF(train_filepaths,test_filepath,target_feature,fg,r,rd,imp,method,matcher,case_num):
    # eta = 0.7;
    num_trees_faidra_list = [20];
    for num_trees_faidra in num_trees_faidra_list:
        print("Number of trees =", num_trees_faidra);
        y_scores = [];
        y_preds = [];
        ll = [];
        print("");
        loss_inters = [];
        for num_rounds in range(0,num_trees_faidra):
            print("");
            print("Round ", str(num_rounds));
            j = 0;
            shs = [];
            X_dists = [];
            loss_inter = [];
            for filepath in train_filepaths:
                print("Reading training data from", filepath);
                X_dist_temp = pd.read_excel(filepath);
                # c = np.size(X_dist_temp,0);
                X_dist_temp.replace('?', np.nan, inplace=True);
                # X_dist_temp = X_dist_temp.iloc[np.random.choice(np.array(X_dist_temp.index),int(c*0.5))];
                
                if(imp == 1):
                    X_dist_temp.fillna(X_dist_temp.median(skipna=True), inplace=True);
                    X_dist_temp.replace(np.nan, 0, inplace=True);
                else:
                    X_dist_temp.replace(np.nan, 0, inplace=True);
                Y_dist_temp = X_dist_temp.iloc[:,target_feature];
                print("Number of MALTs =", np.sum(Y_dist_temp));
                
                # print("Computing downsampling ratio");
                # cohorts.append(train_filepaths[j-1].split('/')[1].split('.')[0]);
                # MALTs.append(np.sum(Y_dist_temp));
                # nonMALTs.append(len(Y_dist_temp)-np.sum(Y_dist_temp));
                features = list(X_dist_temp.columns);
                a = np.array(np.where(Y_dist_temp == 0));
                f = r*np.int(np.sum(Y_dist_temp));
                
                # print("Checking downsampling ratio");
                if(f > np.size(a,1)-1):
                    f = np.size(a,1)-1;  
                
                # print("Searching for controls");
                if(matcher == 1):
                    #make subgroup analysis
                    gender = X_dist_temp.iloc[:,1];
                    ageSS = X_dist_temp.iloc[:,2];
                    dd = X_dist_temp.iloc[:,3];
                    
                    L = np.where(X_dist_temp.iloc[:,target_feature] == 1)[0];
                    NL = np.where(X_dist_temp.iloc[:,target_feature] == 0)[0];
                    
                    gender_L = gender.iloc[L];
                    gender_NL = gender.iloc[NL];
                    
                    ageSS_L = ageSS.iloc[L];
                    ageSS_NL = ageSS.iloc[NL];
                    
                    dd_L = dd.iloc[L];
                    dd_NL = dd.iloc[NL];
                    
                    while(1):
                        # print("Searching for matching populations...");
                        # f_ind = np.array(np.floor(np.linspace(0,np.size(a,1)-1,f)), dtype=int);
                        f_ind = random.sample(range(1, np.size(NL)), int(np.floor(np.size(L)*r)));
                        gender_NL_rand = gender_NL.iloc[f_ind];
                        ageSS_NL_rand = ageSS_NL.iloc[f_ind];
                        dd_NL_rand = dd_NL.iloc[f_ind];
                        
                        # [_, p1] = ranksums(gender_L, gender_NL_rand);
                        [_, p2] = ranksums(ageSS_L, ageSS_NL_rand);
                        [_, p3] = ranksums(dd_L, dd_NL_rand);
                        
                        T_L_males = len(np.where(gender_L == 0)[0]);
                        T_L_females = len(np.where(gender_L == 1)[0]);
                        T_NL_males = len(np.where(gender_NL_rand == 0)[0]);
                        T_NL_females = len(np.where(gender_NL_rand == 1)[0]);
                        
                        if((T_L_males < 5)|(T_L_females < 5)|(T_NL_males < 5)|(T_NL_females < 5)):
                            p1 = sp.fisher_exact([[T_L_males, T_L_females],[T_NL_males, T_NL_females]])[1]; 
                        else:
                            p1 = sp.chi2_contingency([[T_L_males, T_L_females],[T_NL_males, T_NL_females]])[1];
                        
                        if(p1 > 0.05)&(p2 > 0.05)&(p3 > 0.05):
                            print("Found matched population");
                            break;
                else:
                    f_ind = np.array(np.floor(np.linspace(0,np.size(a,1)-1,f)), dtype=int);

                # L = np.where(X_dist_temp.iloc[:,target_feature] == 1)[0];
                # NL = np.where(X_dist_temp.iloc[:,target_feature] == 0)[0];
                
                #drop columns from training dataset
                X_dist_temp.drop(columns=['Cohort', 
                                          X_dist_temp.columns[target_feature]], 
                                 inplace=True);
                
                # print(X_dist_temp.columns);
                # print("Applying downsampling");
                # while(1):
                #     f_ind = random.sample(range(1, np.size(NL)), int(np.floor(np.size(L)*r)));
                
                ####################################################################
                b = a[:,f_ind];
                L_b = np.where(Y_dist_temp == 1);
                bb = np.concatenate((np.array(b),np.array(L_b)),axis=None);
                X_dist_t = X_dist_temp.iloc[bb,:];
                Y_dist_t = Y_dist_temp[bb];
                ####################################################################
                
                # ##############################CAUTION#############################
                # c = np.size(X_dist_t,0);
                # # rchoice = np.random.choice(np.array(X_dist_t.index),int(c*0.5));
                # rchoice = np.random.choice(np.array(range(c)),int(c*0.8));
                # X_dist = X_dist_t.iloc[rchoice,:];
                # Y_dist = Y_dist_t.iloc[rchoice];
                # ##################################################################
                
                X_dist = copy.deepcopy(X_dist_t);
                Y_dist = copy.deepcopy(Y_dist_t);
                
                # print("Creating XGBoost schema");
                X_dist_xgb = xgb.DMatrix(X_dist, Y_dist);
                
                # print("Initiating incremental training");
                # evals_result_inter = {}
                if (j == 0):
                    d = 0.2;
                    params = {'booster':'dart', 
                              # 'objective':'binary:logistic',
                              'rate_drop':d,
                              'eval_metric':'logloss',
                              };
                    d_DF = pd.DataFrame(data={'delta': [d]});
                    d_DF.to_excel('results/v3/MALT/delta_temp.xlsx');
                    model1 = xgb.train(params, 
                                       X_dist_xgb,
                                       obj = my_obj);
                    loss_inter.append(model1.eval(X_dist_xgb));
                else:
                    # params.update({'process_type': 'update', 
                    #                'updater': 'prune', 
                    #                'refresh_leaf': True});
                    model1 = xgb.train(params, 
                                       X_dist_xgb,
                                       xgb_model=model1);
                    loss_inter.append(model1.eval(X_dist_xgb));
                    del X_dist_xgb, Y_dist;
                j += 1;
                # shs.append(shap.TreeExplainer(model1).shap_values(X_dist));
                # X_dists.append(X_dist);
            loss_inters.append(loss_inter);    
            # print("The incremental training process is over");
            print("");
            
            # print("Applying feature selection");
            best_features = [];
            fscores = [];
            if(method == 1):
                features = list(X_dist_temp.columns);
                fscore = model1.get_fscore();
                xgb_features = list(fscore.keys());
                features_ori = list(X_dist_temp.columns);
                both = set(features_ori).intersection(xgb_features);
                xgb_features_ind = [features_ori.index(x) for x in both];
                xgb_features_ind = np.array(xgb_features_ind, dtype=int);
                bf = [features[x] for i,x in enumerate(xgb_features_ind)];
                best_features.append(bf);
                fscores.append(fscore);
            else:
                best_features.append('no');
                fscores.append('no');
            
            print("Testing on", test_filepath);
            T_temp = pd.read_excel(test_filepath);
            T_temp.replace('?', np.nan, inplace=True);
            if(imp == 1):
                T_temp.fillna(T_temp.median(skipna=True), inplace=True);
                T_temp.replace(np.nan, 0, inplace=True);
            else:
                T_temp.replace(np.nan, 0, inplace=True);
            y_T_temp = T_temp.iloc[:,target_feature];
            T_temp.drop(columns=['Cohort', T_temp.columns[target_feature]], inplace=True);
            
            # print("Computing statistics");
            # cohorts.append(test_filepath.split('/')[1].split('.')[0]);
            # MALTs.append(np.sum(y_T_temp));
            # nonMALTs.append(len(y_T_temp)-np.sum(y_T_temp));
            
            # print("Creating XGBoost schema");
            y_T = y_T_temp;
            T = T_temp;
            X_test_xgb = xgb.DMatrix(T, y_T);
            
            # print("Model evaluation");
            y_pred1 = model1.predict(X_test_xgb);
            y_score1 = (y_pred1 >= 0.5)*1;
            acc1 = accuracy_score(y_T,y_score1);
            sensitivity1 = np.sum((y_score1 == 1) & (y_T == 1)) / np.sum(y_T);
            [TN, FP, FN, TP] = metrics.confusion_matrix(y_T, y_score1).ravel();
            specificity1 = TN/(TN+FP);
            [fpr1, tpr1, thresholds] = roc_curve(y_T, y_pred1);
            roc_auc1 = auc(fpr1, tpr1);
            
            print("Accuracy = ", str(acc1));
            print("Sensitivity = ", str(sensitivity1));
            print("Specificity = ", str(specificity1));
            print("AUC = ", str(roc_auc1));
            
            time.sleep(1);
            
            y_scores.append(y_score1);
            y_preds.append(y_pred1);
            ll.append(skl_metrics.log_loss(y_T,y_pred1));
        # print("Number of rounds have been terminated");
        print("");
        
        print("Dumping trees with log loss less than the median log loss");
        s_trees = np.where(np.array(ll) < np.mean(ll))[0];
        if(s_trees is not None)&(len(s_trees) != 0):
            print("Dumping ", str(len(s_trees)), " trees out of ", str(num_trees_faidra));
            y_preds = [y_preds[x] for i,x in enumerate(s_trees)];
            y_scores = [y_scores[x] for i,x in enumerate(s_trees)];
        else:
            print("No dump is necessary");
        
        print("Applying majority voting to derive the final predictions from the remaining trees");
        y_score_final = [];
        for m in range(0,len(y_score1)):
            score = 0;
            for n in range(0,np.size(y_scores,0)):
                score = score + y_scores[n][m];
            
            if(score > np.size(y_scores,0)/2):
                y_score_final.append(1);
            else:
                y_score_final.append(0);
            
        y_pred_final = [];
        y_pred_final.append(np.mean(y_preds,axis=0));
        
        # print(np.array(y_score_final));
        # print("");
        # print(np.array(y_pred_final[0]));
        
        [acc, sens, spec, recall, prec, roc_auc, 
         tpr_final_t, fpr_final_t] = compute_eval_measures_dist(model1, 
                                                                T, 
                                                                y_T, 
                                                                np.array(y_score_final), 
                                                                np.array(y_pred_final[0]));
        fpr_final = np.linspace(0, 1, 100);
        tpr_final = np.interp(fpr_final, fpr_final_t, tpr_final_t);
        tpr_final[0] = 0;
        # AUC = skl_metrics.auc(fpr_final, tpr_final);
        # aucs.append(roc_auc);
        
        sensitivity = np.sum((np.array(y_score_final) == 1) & (y_T == 1)) / np.sum(y_T);
        
        print("");
        print("Length of y_score1", str(len(y_score1)));
        print("Length of y_preds", str(len(y_preds)));
        print("Length of y_preds[0]", str(len(y_preds[0])));
        print("Length of y_pred_final", str(len(y_pred_final)));
        print("Length of y_pred_final[0]", str(len(y_pred_final[0])));
        print("Accuracy = ", str(acc));
        print("Precision = ", str(prec));
        print("Sensitivity 1 = ", str(sensitivity));
        print("Sensitivity 2 = ", str(sens));
        print("Recall = ", str(recall));
        print("Specificity = ", str(spec));
        print("AUC = ", str(roc_auc));
        print("");
        
        time.sleep(2);
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    plt.plot(fpr_final,
             tpr_final,
             # label=r'ROC FHBF, start = %s, end = %s (AUC = %s)' % (train_filepaths[0].split('/')[2].split('.')[0], 
             #                                                       test_filepath.split('/')[2].split('.')[0], 
             #                                                       np.around(roc_auc,3)),  
             label=r'FHBF (AUC = %s)' % (np.around(roc_auc,2)),  
             lw=2, 
             alpha=.8);
    # plt.title('number of FHBF rounds = ' + str(num_trees_faidra));
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
    plt.xlabel('FPR (1-specificity)');
    plt.ylabel('TPR');
    plt.legend();
    plt.show();
        
    return [loss_inters, X_dists, shs, acc, sensitivity, spec, recall, prec, roc_auc];


def apply_ic(train_filepaths,test_filepath,target_feature,fg,r,rd,imp,method,matcher):
    j = 1;
    MALTs = [];
    cohorts = [];
    nonMALTs = [];
    shs = [];
    X_dists = [];
    loss_inter = [];
    for filepath in train_filepaths:
        print("Training on", train_filepaths[j-1]);
        
        X_dist_temp = pd.read_excel(filepath);
        X_dist_temp.replace('?', np.nan, inplace=True);
        
        if(imp == 1):
            X_dist_temp.fillna(X_dist_temp.median(skipna=True), inplace=True);
            X_dist_temp.replace(np.nan, 0, inplace=True);
        else:
            X_dist_temp.replace(np.nan, 0, inplace=True);
        
        Y_dist_temp = X_dist_temp.iloc[:,target_feature];
        # X_dist_temp.drop(columns=['Cohort', X_dist_temp.columns[target_feature]], inplace=True);
        
        print("Number of MALTs =", np.sum(Y_dist_temp));
        cohorts.append(train_filepaths[j-1].split('/')[1].split('.')[0]);
        MALTs.append(np.sum(Y_dist_temp));
        nonMALTs.append(len(Y_dist_temp)-np.sum(Y_dist_temp));
        features = list(X_dist_temp.columns);
        a = np.array(np.where(Y_dist_temp == 0));
        f = r*np.int(np.sum(Y_dist_temp));
        
        if(f > np.size(a,1)-1):
            f = np.size(a,1)-1;  
        
        print("Searching for controls");
        if(matcher == 1):
            gender = X_dist_temp.iloc[:,1];
            ageSS = X_dist_temp.iloc[:,2];
            dd = X_dist_temp.iloc[:,3];
            
            L = np.where(X_dist_temp.iloc[:,target_feature] == 1)[0];
            NL = np.where(X_dist_temp.iloc[:,target_feature] == 0)[0];
            
            gender_L = gender.iloc[L];
            gender_NL = gender.iloc[NL];
            ageSS_L = ageSS.iloc[L];
            ageSS_NL = ageSS.iloc[NL];
            dd_L = dd.iloc[L];
            dd_NL = dd.iloc[NL];
            
            while(1):
                # print("Searching for matching populations...");
                # f_ind = np.array(np.floor(np.linspace(0,np.size(a,1)-1,f)), dtype=int);
                f_ind = random.sample(range(1, np.size(NL)), int(np.floor(np.size(L)*r)));
                gender_NL_rand = gender_NL.iloc[f_ind];
                ageSS_NL_rand = ageSS_NL.iloc[f_ind];
                dd_NL_rand = dd_NL.iloc[f_ind];
                
                # [_, p1] = ranksums(gender_L, gender_NL_rand);
                [_, p2] = ranksums(ageSS_L, ageSS_NL_rand);
                [_, p3] = ranksums(dd_L, dd_NL_rand);
                
                T_L_males = len(np.where(gender_L == 0)[0]);
                T_L_females = len(np.where(gender_L == 1)[0]);
                T_NL_males = len(np.where(gender_NL_rand == 0)[0]);
                T_NL_females = len(np.where(gender_NL_rand == 1)[0]);
                
                if((T_L_males < 5)|(T_L_females < 5)|(T_NL_males < 5)|(T_NL_females < 5)):
                    p1 = sp.fisher_exact([[T_L_males, T_L_females],[T_NL_males, T_NL_females]])[1]; 
                else:
                    p1 = sp.chi2_contingency([[T_L_males, T_L_females],[T_NL_males, T_NL_females]])[1];
                
                if(p1 > 0.05)&(p2 > 0.05)&(p3 > 0.05):
                    print("Found matched population");
                    break;
        else:
            f_ind = np.array(np.floor(np.linspace(0,np.size(a,1)-1,f)), dtype=int);
            
        X_dist_temp.drop(columns=['Cohort', X_dist_temp.columns[target_feature]], inplace=True);
        
        b = a[:,f_ind];
        L_b = np.where(Y_dist_temp == 1);
        bb = np.concatenate((np.array(b),np.array(L_b)),axis=None);
        X_dist = X_dist_temp.iloc[bb,:];
        Y_dist = Y_dist_temp[bb];
        
        if(method == 1):
            X_dist_xgb = xgb.DMatrix(X_dist, Y_dist);
        
        if(method == 1):
            if(j == 1):
                #training
                if(rd == 0):
                    # params = {'objective': 'binary:logistic', 
                    #           'booster':'gbtree', 
                    #           'eval_metric':'logloss'};
                    params = {'booster':'gbtree',
                              'objective': 'binary:logistic',
                              'eval_metric':'logloss'
                              };
                    dlr = xgb.train(params, 
                                    X_dist_xgb);
                    # print(dlr.eval(X_dist_xgb))
                    # loss_inter.append(evals_result['eval']['logloss']);
                    loss_inter.append(dlr.eval(X_dist_xgb));
                else:
                    d = rd/10;
                    # d_DF = pd.DataFrame(data={'delta': [d]});
                    # d_DF.to_excel('results/v2/MALT/delta_temp.xlsx');               
                    params = {'booster':'dart', 
                              'objective': 'binary:logistic',
                              'eval_metric':'logloss',
                              'rate_drop':d};
                    # params = {'booster':'dart', 'rate_drop':np.log(1/1+np.exp(-d))};
                    dlr = xgb.train(params, 
                                    X_dist_xgb);
                    loss_inter.append(dlr.eval(X_dist_xgb));
            else:
                # params.update({'process_type': 'update', 
                #                'updater': 'prune', 
                #                'refresh_leaf': True});
                dlr = xgb.train(params, 
                                X_dist_xgb,
                                xgb_model=dlr);
                loss_inter.append(dlr.eval(X_dist_xgb));
            if(rd == 0):
                shs.append(shap.TreeExplainer(dlr).shap_values(X_dist));
                X_dists.append(X_dist);
            j += 1;
        elif(method == 2):
            print("Mphke MNB gia j =", str(j));
            if(j == 1):
                dlr = naive_bayes.MultinomialNB();
            dlr.partial_fit(X_dist,Y_dist,np.unique(Y_dist));
            j += 1;
        elif(method == 3):
            print("Mphke SVM gia j =", str(j));
            if(j == 1):
                dlr = linear_model.SGDClassifier(penalty = 'l2', 
                                                 warm_start=True, 
                                                 shuffle=False);
            dlr.partial_fit(X_dist,Y_dist,np.unique(Y_dist));
            j += 1;
        elif(method == 4):
            print("Mphke MLP gia j =", str(j));
            if(j == 1):
                dlr = MLPClassifier(warm_start=True,
                                    hidden_layer_sizes=(380,),
                                    solver='sgd',
                                    random_state=0);
            dlr.partial_fit(X_dist,Y_dist,np.unique(Y_dist));
            j += 1;
            
    best_features = [];
    fscores = [];
    if(method == 1):
        #feature selection
        features = list(X_dist_temp.columns);
        fscore = dlr.get_fscore();
        xgb_features = list(fscore.keys());
        # print("Features:", xgb_features);
        features_ori = list(X_dist_temp.columns);
        both = set(features_ori).intersection(xgb_features);
        xgb_features_ind = [features_ori.index(x) for x in both];
        xgb_features_ind = np.array(xgb_features_ind, dtype=int);
        bf = [features[x] for i,x in enumerate(xgb_features_ind)];
        # print("Best features:", bf);
        best_features.append(bf);
        fscores.append(fscore);
        print(fscores);
    else:
        best_features.append('no');
        fscores.append('no');
            
    print("Testing on", test_filepath);
    T_temp = pd.read_excel(test_filepath);
    T_temp.replace('?', np.nan, inplace=True);
    
    if(imp == 1):
        T_temp.fillna(T_temp.median(skipna=True), inplace=True);
        T_temp.replace(np.nan, 0, inplace=True);
    else:
        T_temp.replace(np.nan, 0, inplace=True);
    
    y_T_temp = T_temp.iloc[:,target_feature];
    T_temp.drop(columns=['Cohort', T_temp.columns[target_feature]], inplace=True);
    
    cohorts.append(test_filepath.split('/')[1].split('.')[0]);
    MALTs.append(np.sum(y_T_temp));
    nonMALTs.append(len(y_T_temp)-np.sum(y_T_temp));
    
    y_T = y_T_temp;
    T = T_temp;
    
    # print("Number of MALTs =", np.sum(y_T));
    if(method == 1):
        X_test_xgb = xgb.DMatrix(T, y_T);
    
        #evaluation
        y_pred1 = dlr.predict(X_test_xgb);
        y_score1 = (y_pred1 >= 0.5)*1;
        acc1 = accuracy_score(y_T,y_score1);
        rec1 = metrics.recall_score(y_T,y_score1);
        prec1 = metrics.f1_score(y_T,y_score1);
        f1 = metrics.precision_score(y_T,y_score1);
        sensitivity1 = np.sum((y_score1 == 1) & (y_T == 1)) / np.sum(y_T);
        [TN, FP, FN, TP] = metrics.confusion_matrix(y_T, y_score1).ravel();
        specificity1 = TN/(TN+FP);
        [fpr1, tpr1, thresholds] = roc_curve(y_T, y_pred1);
        roc_auc1 = auc(fpr1, tpr1);
    else:
        #evaluation
        y_pred1 = dlr.predict(T);
        y_score1 = (y_pred1 >= 0.5)*1;
        acc1 = accuracy_score(y_T,y_score1);
        rec1 = metrics.recall_score(y_T,y_score1);
        prec1 = metrics.f1_score(y_T,y_score1);
        f1 = metrics.precision_score(y_T,y_score1);
        sensitivity1 = np.sum((y_score1 == 1) & (y_T == 1)) / np.sum(y_T);
        [TN, FP, FN, TP] = metrics.confusion_matrix(y_T, y_score1).ravel();
        specificity1 = TN/(TN+FP);
        [fpr1, tpr1, thresholds] = roc_curve(y_T, y_pred1);
        roc_auc1 = auc(fpr1, tpr1);        

    fpr = fpr1;
    tpr = tpr1;

    if(fg == 1):
        if((method == 1)&(sensitivity1 > 0.5)):
            print('Accuracy: %.4f' %acc1);
            print('Recall: %.2f' %rec1);
            print('F1-score: %.4f' %f1);
            print('Precision: %.4f' %prec1);
            print('AUC: %.4f' %roc_auc1);
            print('Sensitivity: %.4f' %sensitivity1);
            print('Specificity: %.4f' %specificity1);
            print(metrics.confusion_matrix(y_T,y_score1));
            
            tprs = [];
            mean_fpr = np.linspace(0, 1, 100);
            tprs.append(interp(mean_fpr, fpr, tpr));
            tprs[-1][0] = 0.0;
            
            mean_tpr = np.mean(tprs, axis=0);
            mean_tpr[-1] = 1.0;
            mean_auc = auc(mean_fpr, mean_tpr);
            if(rd != 0):
                new_palette = sns.color_palette("bone")[0:4][::-1]
                new_palette[-4] = sns.color_palette("bone")[3]
                new_palette[-3] = sns.color_palette("Blues")[2]
                new_palette[-2] = sns.color_palette("Blues")[5]
                sns.set_palette(sns.color_palette(new_palette))
                sns.color_palette(new_palette)
                sns.set_style("white")
                plt.plot(mean_fpr, mean_tpr, label=r'FDART, rate_drop = %0.2f (AUC = %0.2f)' % (d, mean_auc), lw=2, alpha=.5);
            else:
                new_palette = sns.color_palette("bone")[0:4][::-1]
                new_palette[-4] = sns.color_palette("bone")[3]
                new_palette[-3] = sns.color_palette("Blues")[2]
                new_palette[-2] = sns.color_palette("Blues")[5]
                sns.set_palette(sns.color_palette(new_palette))
                sns.color_palette(new_palette)
                sns.set_style("white")
                plt.plot(mean_fpr, mean_tpr, label=r'FGBT (AUC = %0.2f)' % (mean_auc), lw=2, alpha=.5);          
            plt.xlim([-0.05, 1.05]);
            plt.ylim([-0.05, 1.05]);
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.legend(loc="lower right");
            plt.show();       
        elif((method == 2)&(sensitivity1 > 0.5)):
            print('Accuracy: %.4f' %acc1);
            print('Recall: %.2f' %rec1);
            print('F1-score: %.4f' %f1);
            print('Precision: %.4f' %prec1);
            print('AUC: %.4f' %roc_auc1);
            print('Sensitivity: %.4f' %sensitivity1);
            print('Specificity: %.4f' %specificity1);
            print(metrics.confusion_matrix(y_T,y_score1));
            
            tprs = [];
            mean_fpr = np.linspace(0, 1, 100);
            tprs.append(interp(mean_fpr, fpr, tpr));
            tprs[-1][0] = 0.0;
            
            new_palette = sns.color_palette("bone")[0:4][::-1]
            new_palette[-4] = sns.color_palette("bone")[3]
            new_palette[-3] = sns.color_palette("Blues")[2]
            new_palette[-2] = sns.color_palette("Blues")[5]
            sns.set_palette(sns.color_palette(new_palette))
            sns.color_palette(new_palette)
            sns.set_style("white")
            
            mean_tpr = np.mean(tprs, axis=0);
            mean_tpr[-1] = 1.0;
            mean_auc = auc(mean_fpr, mean_tpr);
            plt.plot(mean_fpr, mean_tpr, label='FNB (AUC = %0.2f)' % (mean_auc), lw=2, alpha=.5);          
            plt.xlim([-0.05, 1.05]);
            plt.ylim([-0.05, 1.05]);
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.legend(loc="lower right");
            plt.show();
        elif((method == 3)&(sensitivity1 > 0.5)&(sensitivity1 != 1)):
            print('Accuracy: %.4f' %acc1);
            print('Recall: %.2f' %rec1);
            print('F1-score: %.4f' %f1);
            print('Precision: %.4f' %prec1);
            print('AUC: %.4f' %roc_auc1);
            print('Sensitivity: %.4f' %sensitivity1);
            print('Specificity: %.4f' %specificity1);
            print(metrics.confusion_matrix(y_T,y_score1));
            
            tprs = [];
            mean_fpr = np.linspace(0, 1, 100);
            tprs.append(interp(mean_fpr, fpr, tpr));
            tprs[-1][0] = 0.0;
            
            new_palette = sns.color_palette("bone")[0:4][::-1]
            new_palette[-4] = sns.color_palette("bone")[3]
            new_palette[-3] = sns.color_palette("Blues")[2]
            new_palette[-2] = sns.color_palette("Blues")[5]
            sns.set_palette(sns.color_palette(new_palette))
            sns.color_palette(new_palette)
            sns.set_style("white")
            
            mean_tpr = np.mean(tprs, axis=0);
            mean_tpr[-1] = 1.0;
            mean_auc = auc(mean_fpr, mean_tpr);
            plt.plot(mean_fpr, mean_tpr, label='FSGD (hinge loss) (AUC = %0.2f)' % (mean_auc), lw=2, alpha=.5);          
            plt.xlim([-0.05, 1.05]);
            plt.ylim([-0.05, 1.05]);
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.legend(loc="lower right");
            plt.show();
        elif((method == 4)&(sensitivity1 > 0.5)&(sensitivity1 != 1)):    
            print('Accuracy: %.4f' %acc1);
            print('Recall: %.2f' %rec1);
            print('F1-score: %.4f' %f1);
            print('Precision: %.4f' %prec1);
            print('AUC: %.4f' %roc_auc1);
            print('Sensitivity: %.4f' %sensitivity1);
            print('Specificity: %.4f' %specificity1);
            print(metrics.confusion_matrix(y_T,y_score1));
            
            tprs = [];
            mean_fpr = np.linspace(0, 1, 100);
            tprs.append(interp(mean_fpr, fpr, tpr));
            tprs[-1][0] = 0.0;
            
            new_palette = sns.color_palette("bone")[0:4][::-1]
            new_palette[-4] = sns.color_palette("bone")[3]
            new_palette[-3] = sns.color_palette("Blues")[2]
            new_palette[-2] = sns.color_palette("Blues")[5]
            sns.set_palette(sns.color_palette(new_palette))
            sns.color_palette(new_palette)
            sns.set_style("white")
            
            mean_tpr = np.mean(tprs, axis=0);
            mean_tpr[-1] = 1.0;
            mean_auc = auc(mean_fpr, mean_tpr);
            plt.plot(mean_fpr, mean_tpr, label='FANN (AUC = %0.2f)' % (mean_auc), lw=2, alpha=.5);          
            plt.xlim([-0.05, 1.05]);
            plt.ylim([-0.05, 1.05]);
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.legend(loc="lower right");
            plt.show();     
    return [loss_inter, X_dists, shs, acc1, sensitivity1, rec1, prec1, f1, roc_auc1, fpr1, tpr1, dlr, features, specificity1, MALTs, cohorts, nonMALTs, best_features, fscores];

#CONTROL PANEL
methods = [1,2,3]; #incremental ML methods
r = 1; #downsampling ratio
fg = 1; #printing option
imp = 1; #mehod for imputation, 1 = median
matcher = 1; #method for population matching, 1 = random
rds = [0,1,2]; #delta steps for loss function
target_feature = 31; #target feature ID
proceed_plot = 1; #whether to plot losses or not

#with 0 imp
filepaths1 = ['UOA', 'UNIPI', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'HUA', 
              'UOI', 'UU', 'UNIRO', 'UMCU', 'UBO', 'AOUD'];

filepaths2 = ['UNIPI', 'UOA', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 
              'UOI', 'UU', 'UNIRO', 'UMCU', 'UBO', 'AOUD', 'HUA'];

filepaths3 = ['AOUD', 'UNIPI', 'UOA', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 
              'UOI', 'UU', 'UNIRO', 'UMCU', 'UBO', 'HUA'];

filepaths4 = ['UOA', 'UNIPI', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'AOUD', 
              'UOI', 'UU', 'UNIRO', 'UMCU', 'UBO', 'HUA'];

#with MD imp
filepaths5 = ['UOA', 'UNIPI', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'HUA', 'UOI', 'UU', 
             'UNIRO', 'UMCU', 'UBO', 'AOUD'];

filepaths6 = ['AOUD', 'UOA', 'UNIPI', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'UOI', 'UU', 
             'UNIRO', 'UMCU', 'UBO', 'HUA'];

filepaths7 = ['AOUD', 'UOA', 'UNIPI', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'HUA', 'UOI',
             'UNIRO', 'UMCU', 'UBO', 'UU'];

#some extras with MD imp
filepaths8 = ['AOUD', 'UOA', 'UNIPI', 'UNEW', 'PARIS', 'UoB', 'UNIVAQ', 'HUA', 'UOI', 
             'UNIRO', 'UMCU', 'UBO', 'UU', 'UNIPG'];

filepaths9 = ['UOA', 'AOUD', 'UNIPI', 'UNEW', 'PARIS', 'UoB', 'UNIVAQ', 'HUA', 'UOI', 
             'UNIRO', 'UMCU', 'UBO', 'UU', 'UNIPG'];

filepaths10 = ['UOA', 'AOUD', 'UNIPI', 'UNEW', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'UOI',
               'UNIRO', 'UMCU', 'UBO', 'UU', 'HUA'];

#definition of the filepaths
filepathss = [];
filepathss.append(filepaths1);
filepathss.append(filepaths2);
filepathss.append(filepaths3);
filepathss.append(filepaths4);
filepathss.append(filepaths5);
filepathss.append(filepaths6);
filepathss.append(filepaths7);
filepathss.append(filepaths8);
filepathss.append(filepaths9);
filepathss.append(filepaths10);

coh_name = ['IDIPAPS', 'UNIPG', 'PARIS', 'UoB', 'UNIVAQ', 'ULB', 'HUA', 'UMCG',
            'UiB', 'UOI', 'UU', 'UNIRO', 'QMUL', 'UMCU', 'MHH', 'UNIPI', 'CUMB',
            'UBO', 'UOA', 'AOUD', 'UNEW', 'ASSESS'];
coh_ID = np.array([2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 28, 31, 33]);

#FHBF vs FGBT vs FDART for multiple dropout rates
c1 = 1;
case = [];
accs_total_2 = [];
sens_total_2 = [];
specs_total_2 = [];
aucs_total_2 = [];
recs_total_2 = [];
shs_total_2 = [];
X_dists_total_2 = [];
algos = [];
lossss = [];
for filepaths in filepathss:
    if(c1 == 5):
        imp = 1;
    else:
        imp = 0;
    
    if(c1 == 5):
        imp = 1;
    else:
        imp = 0;
        
    ss = [coh_name.index(x) for x in filepaths];
    train_filepaths = [];
    test_filepath = [];
    print("");
    combo = ["Combination: i = " + str(ss)];
    print(combo);
    for c in ss[0:-1]:
        train_filepaths.append('data/v2/FINAL_MALT/'+coh_name[c]+'.xlsx');
    test_filepath.append('data/v2/FINAL_MALT/'+coh_name[ss[-1]]+'.xlsx');
    
    print("Training on->", train_filepaths);
    print("Testing on->", test_filepath);
    
    accs = [];
    sens = [];
    recs = [];
    precs = [];
    f1s = [];
    rocs = [];
    specs = [];
    aucs = [];
    best_features = [];
    f_scores = [];
    losss = [];
    
    for rd in rds:
        [loss, X_dists, shs, acc1, sensitivity1, rec1, prec1, f1, roc_auc1, 
         fpr1, tpr1, dlr, features, specificity1, MALTs, cohorts, nonMALTs, 
         best_features1, fscores] = apply_ic(train_filepaths,
                                            test_filepath[0],
                                            target_feature,
                                            fg,
                                            r,
                                            rd,
                                            imp,
                                            methods[0],
                                            matcher);
                                                           
        accs.append(acc1);
        sens.append(sensitivity1);
        precs.append(prec1);
        recs.append(rec1);
        f1s.append(f1);
        rocs.append(roc_auc1);
        specs.append(specificity1);
        aucs.append(roc_auc1);
        best_features.append(best_features1);
        f_scores.append(fscores);
        losss.append(loss);
        
        accs_total_2.append(acc1);
        specs_total_2.append(specificity1);
        sens_total_2.append(sensitivity1);
        aucs_total_2.append(roc_auc1);
        recs_total_2.append(rec1);
        shs_total_2.append(shs);
        X_dists_total_2.append(X_dists);
        case.append('case'+str(c1));
        algos.append('FGBT_rd_'+str(rd));

    [loss, X_dists, shs, acc, sens, spec, recall, _, roc_auc] = FHBF(train_filepaths,
                                                                test_filepath[0],
                                                                target_feature,
                                                                fg,
                                                                r,
                                                                0,
                                                                imp,
                                                                2,
                                                                matcher,
                                                                0);
    
    accs_total_2.append(acc);
    specs_total_2.append(spec);
    sens_total_2.append(sens);
    aucs_total_2.append(roc_auc);
    recs_total_2.append(recall);
    shs_total_2.append(shs);
    losss.append(loss);
    X_dists_total_2.append(X_dists);
    case.append('case'+str(c1));
    algos.append('FHBF');
    lossss.append(losss);
    
    # plt.title('ROC curves (starting cohort:'+coh_name[ss[0]]+', testing cohort:'+coh_name[ss[-1]]+')');
    plt.savefig('results/v3/MALT/case_'+str(c1)+'_FHBF_vs_FGBT_vs_FDART.jpg', 
                dpi=600, 
                bbox_inches="tight");
    plt.close();
    c1 += 1;
    
R = pd.DataFrame(data={'case':case,
                       'algo':algos,
                       'accuracy':accs_total_2,
                       'sens':sens_total_2,
                       'spec':specs_total_2,
                       'auc':aucs_total_2,
                       'rec':recs_total_2});
R.to_excel('results/v3/MALT_results_total.xlsx',
           index=False);

################################ losss indexing ###############################
case_indices = np.array([0,4,6]); #cases 1,3,5,6,7 or 1,5,6,7 or 1,3,5,7 or 1,5,7
losssss = [lossss[i] for i in case_indices]
# losssss = copy.deepcopy(lossss);
fg_replace = 4;
###############################################################################

if(fg_replace == 0):
    file1 = open("results/v3/MALT/Loss_Values.txt", "w")
    file1.write("%s = %s\n" %("dict1", lossss))
    file1.close()
    
if (proceed_plot == 1):
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    sns.set_style("white")
    
    #transform losses to matrices for each cohort across cases
    plt.figure(figsize=[32,22])
    dfloss = [];
    method_ = [];
    cases_ = [];
    cohs_ = [];
    sns.set(font_scale=1.5)
    for k in range(0, len(losssss)):
        fpaths = filepathss[k];
        GBT_loss = losssss[k][0];
        DART_1_loss = losssss[k][1];
        DART_2_loss = losssss[k][2];
        FHBF_loss = losssss[k][3];
        
        GBT_loss = np.array([float(s.split(':')[1]) for s in GBT_loss])
        DART_1_loss = np.array([float(s.split(':')[1]) for s in DART_1_loss])
        DART_2_loss = np.array([float(s.split(':')[1]) for s in DART_2_loss])
        
        FHBF_loss_new = [];
        for i in range(0,len(FHBF_loss)):
            FHBF_loss_new.append(np.array([float(s.split(':')[1]) for s in FHBF_loss[i]]))
        
        FHBF_loss_new_good = [x for i,x in enumerate(FHBF_loss_new) 
                              if np.mean(FHBF_loss_new[i]) <= np.mean(FHBF_loss_new)];
        
        dfloss.append(GBT_loss)
        dfloss.append(DART_1_loss)
        dfloss.append(DART_2_loss)
        dfloss.append(np.nanmean(FHBF_loss_new_good,0))
        
        for j in range(0,len(losssss[0][0])):
            method_.append('FGBT')
            cases_.append('case '+str(k+1))
            cohs_.append(fpaths[j])
            
        for j in range(0,len(losssss[0][0])):
            method_.append('FDART, rate_drop=0.1');
            cases_.append('case '+str(k+1));
            cohs_.append(fpaths[j]);
        
        for j in range(0,len(losssss[0][0])):
            method_.append('FDART, rate_drop=0.2');
            cases_.append('case '+str(k+1));
            cohs_.append(fpaths[j]);
            
        for j in range(0,len(FHBF_loss_new_good[0])):
            method_.append('FHBF');
            cases_.append('case '+str(k+1));
            cohs_.append(fpaths[j]);
        
        ps = [];
        for u in range(0,len(FHBF_loss_new_good)):
            ps.append(np.mean(FHBF_loss_new_good[u]));
        m_ind = np.where(ps == min(ps))[0][0];
        
        plt.subplot(2,5,k+1)
        plt.plot(GBT_loss, 'o-', label='FGBT', alpha=0.7)
        plt.plot(DART_1_loss, 'o-', label='FDART, rate_drop=0.1', alpha=0.7)
        plt.plot(DART_2_loss, 'o-', label='FDART, rate_drop=0.2', alpha=0.7)
        # plt.plot(np.nanmean(FHBF_loss_new_good,0), 'o-', label='FHBF', alpha=0.7)
        plt.plot(FHBF_loss_new_good[m_ind], 'o-', label='FHBF', alpha=0.7)
        plt.xticks(np.array(range(0,len(GBT_loss))),np.array(range(1,len(GBT_loss)+1)))
        plt.xlabel('case '+str(k+1))
        plt.ylabel('')
        plt.ylim(0,12)
        plt.show()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('results/v3/MALT/logloss_cases_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    #barplot of the loss
    dfloss = np.concatenate((dfloss))
    sns.set(font_scale=1.5)
    # sns.set_palette("bone")
    dfloss_df_total = pd.DataFrame(data={'average log loss (training)':dfloss, 'method':method_, 'case':cases_, 'cohs':cohs_})
    plt.figure(figsize=(18,15))
    sns.barplot(data=dfloss_df_total, x='case', y='average log loss (training)', hue='method')
    plt.savefig('results/v3/MALT/logloss_cases_bars_overall_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    #transform losses to matrices for each cohort across cases
    plt.figure(figsize=[32,22])
    for k in range(0, len(losssss)):
        dfloss = [];
        method_ = [];
        cases_ = [];
        
        GBT_loss = losssss[k][0];
        DART_1_loss = losssss[k][1];
        DART_2_loss = losssss[k][2];
        FHBF_loss = losssss[k][3];
        
        GBT_loss = np.array([float(s.split(':')[1]) for s in GBT_loss])
        DART_1_loss = np.array([float(s.split(':')[1]) for s in DART_1_loss])
        DART_2_loss = np.array([float(s.split(':')[1]) for s in DART_2_loss])
        
        FHBF_loss_new = [];
        for i in range(0,len(FHBF_loss)):
            FHBF_loss_new.append(np.array([float(s.split(':')[1]) for s in FHBF_loss[i]]))
        
        FHBF_loss_new_good = [x for i,x in enumerate(FHBF_loss_new) 
                              if np.mean(FHBF_loss_new[i]) <= np.mean(FHBF_loss_new)];
        
        dfloss.append(GBT_loss)
        dfloss.append(DART_1_loss)
        dfloss.append(DART_2_loss)
        dfloss.append(np.nanmean(FHBF_loss_new_good,0))
        
        for j in range(0,len(losssss[0][0])):
            method_.append('FGBT')
            cases_.append('case '+str(k+1))
            
        for j in range(0,len(losssss[0][0])):
            method_.append('FDART, rate_drop=0.1')
            cases_.append('case '+str(k+1))
        
        for j in range(0,len(losssss[0][0])):
            method_.append('FDART, rate_drop=0.2')
            cases_.append('case '+str(k+1))
            
        for j in range(0,len(FHBF_loss_new_good[0])):
            method_.append('FHBF')
            cases_.append('case '+str(k+1))
        
        dfloss_df = pd.DataFrame(data={'average log loss (training)':np.concatenate((dfloss)), 'method':method_, 'case':cases_})
        plt.subplot(2,5,k+1)
        sns.barplot(data=dfloss_df, x='case', y='average log loss (training)', hue='method')
        plt.legend([],[], frameon=False)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.color_palette(new_palette)
    plt.savefig('results/v3/MALT/logloss_cases_bars_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    plt.figure(figsize=[18,12])
    sns.barplot(data=dfloss_df_total, x='cohs', y='average log loss (training)', hue='method')
    plt.savefig('results/v3/MALT/logloss_cohorts_bars_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    #transform losses to matrices for each cohort across cases
    cohs_list = [s for s in filepathss[0]]
    # cohs_list = [s for s in filepathss[0] if (s != 'HUA')&(s != 'UU')&(s != 'UNIRO')&(s != 'AOUD')&(s != 'UNIPG')]
    plt.figure(figsize=[32,22])
    for k in range(0, len(cohs_list)):
        plt.subplot(4,4,k+1)
        W = dfloss_df_total.iloc[np.where(dfloss_df_total['cohs'] == cohs_list[k])[0],:]
        plt.plot(W.iloc[np.where(W['method'] == 'FGBT')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FGBT')
        plt.plot(W.iloc[np.where(W['method'] == 'FDART, rate_drop=0.1')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FDART, rate_drop=0.1')
        plt.plot(W.iloc[np.where(W['method'] == 'FDART, rate_drop=0.2')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FDART, rate_drop=0.2')
        plt.plot(W.iloc[np.where(W['method'] == 'FHBF')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FHBF')
        plt.xticks(np.array(range(0,10)),np.array(range(1,11)))
        plt.ylabel(cohs_list[k])
        plt.xlabel('')
        plt.show()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.color_palette(new_palette)
    plt.savefig('results/v3/MALT/logloss_cohorts_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    #transform losses to matrices for each cohort across cases
    # cohs_list = [s for s in filepathss[0]]
    cohs_list = [s for s in filepathss[0] if (s != 'HUA')&(s != 'UU')&(s != 'UNIRO')&(s != 'AOUD')&(s != 'UNIPG')]
    plt.figure(figsize=[32,22])
    for k in range(0, len(cohs_list)):
        plt.subplot(2,5,k+1)
        W = dfloss_df_total.iloc[np.where(dfloss_df_total['cohs'] == cohs_list[k])[0],:]
        plt.plot(W.iloc[np.where(W['method'] == 'FGBT')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FGBT')
        plt.plot(W.iloc[np.where(W['method'] == 'FDART, rate_drop=0.1')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FDART, rate_drop=0.1')
        plt.plot(W.iloc[np.where(W['method'] == 'FDART, rate_drop=0.2')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FDART, rate_drop=0.2')
        plt.plot(W.iloc[np.where(W['method'] == 'FHBF')[0],0].reset_index(drop=True), 'o-', alpha=0.7, label='FHBF')
        plt.xticks(np.array(range(0,10)),np.array(range(1,11)))
        plt.ylabel(cohs_list[k])
        plt.xlabel('')
        plt.show()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.color_palette(new_palette)
    plt.savefig('results/v3/MALT/logloss_cohorts_no_test_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    #transform losses to matrices for each cohort across cases
    cohs_list = [s for s in filepathss[0]]
    # cohs_list = [s for s in filepathss[0] if (s != 'HUA')&(s != 'UU')&(s != 'UNIRO')&(s != 'AOUD')&(s != 'UNIPG')]
    plt.figure(figsize=[32,22])
    for k in range(0, len(cohs_list)):
        plt.subplot(4,4,k+1)
        W = dfloss_df_total.iloc[np.where(dfloss_df_total['cohs'] == cohs_list[k])[0],:]
        
        sns.barplot(data=W, x='cohs', y='average log loss (training)', hue='method')
        plt.legend([],[], frameon=False)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.color_palette(new_palette)
    plt.savefig('results/v3/MALT/logloss_cohorts_bars_detailed_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    new_palette = sns.color_palette("bone")[0:4][::-1]
    new_palette[-4] = sns.color_palette("bone")[3]
    new_palette[-3] = sns.color_palette("Blues")[2]
    new_palette[-2] = sns.color_palette("Blues")[5]
    sns.set_palette(sns.color_palette(new_palette))
    sns.color_palette(new_palette)
    
    #transform losses to matrices for each cohort across cases
    # cohs_list = [s for s in filepathss[0]]
    cohs_list = [s for s in filepathss[0] if (s != 'HUA')&(s != 'UU')&(s != 'UNIRO')&(s != 'AOUD')&(s != 'UNIPG')]
    plt.figure(figsize=[32,22])
    for k in range(0, len(cohs_list)):
        plt.subplot(2,5,k+1)
        W = dfloss_df_total.iloc[np.where(dfloss_df_total['cohs'] == cohs_list[k])[0],:]
        
        sns.barplot(data=W, x='cohs', y='average log loss (training)', hue='method')
        plt.legend([],[], frameon=False)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.color_palette(new_palette)
    plt.savefig('results/v3/MALT/logloss_cohorts_bars_detailed_no_test_'+str(fg_replace)+'.jpg', dpi=600, bbox_inches='tight')
    plt.close()
