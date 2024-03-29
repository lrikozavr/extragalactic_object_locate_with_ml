# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import os
import sklearn.metrics as skmetrics

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
'''
def max_accuracy_check(path,names):
	max = -1
	max_name = ''
	for name in names:
		if(name == 10130209.0):
			break
		data = pd.read_csv(f"{path}/extragal_custom_{name}_evaluate.csv", header=0, sep=',')
		if(data['Accuracy'].iloc[0] > max):
			max = data['Accuracy'].iloc[0]
			max_name = name
	print(max_name, max)

names = []
for l2 in range(4,16,1):
	for l3 in range(2,16,1):
		for l4 in range(1,16,1):
			names.append(float(str(1) + "0" + str(l2) + "0" + str(l3) + "0" + str(l4)))

max_accuracy_check('ml/eval',names)
'''
'''
data = pd.read_csv('ml/eval/extragal_custom_15015012.0_prob.csv',header=0, sep=',')
precision, recall, thresholds = skmetrics.precision_recall_curve(data['Y'], data['y_prob'])
pr_curve_df = pd.DataFrame({"precision": precision, "recall": recall, 
								"thresholds": np.append(thresholds, 1)})
pr_curve_df = pr_curve_df[pr_curve_df['thresholds'] < 0.99]
max,inde=0,0
for ii in range(len(pr_curve_df)):
	s = math.sqrt(pr_curve_df["recall"].iloc[ii]**2 + pr_curve_df["precision"].iloc[ii]**2)
	if(max < s):
		max = s
		inde = ii
print("thresholds....",pr_curve_df["thresholds"].iloc[inde],"recall....",pr_curve_df["recall"].iloc[inde],"precision....",pr_curve_df["precision"].iloc[inde])
'''
'''
path_save_eval = '/home/lrikozavr/ML_work/des_pro/ml/eval/extragal'
mass_model_name = []
from ml_network import before_ev
max_Ac, max_F1 = 0, 0
min_Ac, min_F1 = 1, 1
name_max_Ac, name_max_F1 = '', ''
name_min_Ac, name_min_F1 = '', ''
for l2 in range(12,16,1):
	for l3 in range(8,12,1):
		for l4 in range(4,8,1):
			sum_Ac, sum_F1 = 0, 0
			name = str(l2) + "n" + str(l3) + "n" + str(l4)
			for i in range(5):
				sub_name = str(i) + "n" + name
				#before_ev(path_save_eval,f'custom_sm_{sub_name}')
				name_col = ['star_cls','gal_cls','qso_cls']
				for cls_name in name_col:
					file_eval = pd.read_csv(f"{path_save_eval}_custom_sm_{sub_name}_{cls_name}_evaluate.csv", header=0, sep=',')
					sum_Ac += file_eval['Accuracy'].iloc[0]/3.
					sum_F1 += file_eval['F1'].iloc[0]/3.

			if( sum_Ac/5. > max_Ac):
				max_Ac = sum_Ac/5.
				name_max_Ac = name
			if( sum_F1/5. > max_F1):
				max_F1 = sum_F1/5.
				name_max_F1 = name
			if( sum_Ac/5. < min_Ac):
				min_Ac = sum_Ac/5.
				name_min_Ac = name
			if( sum_F1/5. < min_F1):
				min_F1 = sum_F1/5.
				name_min_F1 = name


print(name_max_Ac, max_Ac)
print(name_max_F1, max_F1)
print(name_min_Ac, min_Ac)
print(name_min_F1, min_F1)
exit()
'''

#legend_size = 20

#path_save_eval = '/media/kiril/j_08/ML/extragal'
#path_save_eval = 'ml/eval'

#path_classifire = '/home/kiril/github/ML_with_AGN/ML/code/results'
#name_classifire = os.listdir(path_classifire)

#fuzzy_options = ['normal']
#fuzzy_options = ['normal', 'fuzzy_err', 'fuzzy_dist']
#fuzzy_options = ['fuzzy_dist']

#name_sample = ['gal','qso','star']
#name_sample = ['gal']
#name_sample = ['extragal']


def Simpson(a,f):
	n = len(f)
	s=0
	for i in range(1,n):
		s += (abs(a[i]-a[i-1]))*(f[i]+f[i-1])
	s*=0.5
	return s

def eval(y,y_pred,n):
	count = 0
	TP, FP, TN, FN = 0,0,0,0
	Y = 0
	for i in range(n):
		if(y[i]<0.5):
			Y = 0
		if(y[i]>=0.5):
			Y = 1
		if(Y==y_pred[i]):
			count+=1
		if(Y==1):
			if(Y==y_pred[i]):
				TP += 1
			else:
				FP += 1
		if(Y==0):
			if(Y==y_pred[i]):
				TN += 1
			else:
				FN += 1
	Acc = count/n
	pur_a = TP/(TP+FP)
	pur_not_a = TN/(TN+FN)
	com_a = TP/(TP+FN)
	com_not_a = TN/(TN+FP)
	f1 = 2*TP/(2*TP+FP+FN)
	fpr = FP/(TN+FN)
	tnr = TN/(TN+FN)
	bAcc = (TP/(TP+FP)+TN/(TN+FN))/2.
	k = 2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
	mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	BinBs = (FP+FN)/(TP+FP+FN+TN)

	print(np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs]))
	ev = pd.DataFrame([np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs])], 
    columns=['Accuracy','AGN_purity','nonAGN_precision','AGN_completness','nonAGN_completness','F1',
    'FPR','TNR','bACC','K','MCC','BinaryBS'])

	return ev

#def i_need_more_eval(ax1_p, ax2_p, ax3_p, ax1_r, ax2_r, ax3_r, index, name, label, data_general_label):
def i_need_more_eval(path_save_eval, name, label, data_prob):
	fpr, tpr, thresholds = skmetrics.roc_curve(label, data_prob, pos_label=1)
	roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr,
									"thresholds": thresholds})
	roc_curve_df = roc_curve_df[roc_curve_df['thresholds'] < 0.99]									
	roc_curve_df.to_csv(f"{path_save_eval}/{name}_roc.csv", index=False)
	print("ROC_CURVE......",name,":.......",Simpson(np.array(roc_curve_df["fpr"]),np.array(roc_curve_df["tpr"])))
	#print("ROC_CURVE......",name,":.......",Simpson(tpr,fpr))

	precision, recall, thresholds = skmetrics.precision_recall_curve(label, data_prob)
	pr_curve_df = pd.DataFrame({"precision": precision, "recall": recall, 
                                    "thresholds": np.append(thresholds, 1)})
	pr_curve_df = pr_curve_df[pr_curve_df['thresholds'] < 0.99]	
	pr_curve_df.to_csv(f"{path_save_eval}/{name}_pr.csv", index=False)
	print("PR_CURVE.......",name,":.......",Simpson(pr_curve_df["recall"],pr_curve_df["precision"]))
	#print("PR_CURVE.......",name,":.......",Simpson(precision,recall))
	max,inde=0,0
	for ii in range(len(pr_curve_df)):
		s = math.sqrt(pr_curve_df["recall"].iloc[ii]**2 + pr_curve_df["precision"].iloc[ii]**2)
		if(max < s):
			max = s
			inde = ii
	print("thresholds....",pr_curve_df["thresholds"].iloc[inde],"recall....",pr_curve_df["recall"].iloc[inde],"precision....",pr_curve_df["precision"].iloc[inde])

	#ax1_r.plot(roc_curve_df['thresholds'],roc_curve_df['fpr'],c=c[index],label=name)
	#ax2_r.plot(roc_curve_df['thresholds'],roc_curve_df['tpr'],c=c[index],label=name)
	#ax3_r.plot(roc_curve_df['fpr'],roc_curve_df['tpr'],c=c[index],label=name)

	#ax1_p.plot(pr_curve_df['thresholds'],pr_curve_df['precision'],c=c[index],label=name)
	#ax2_p.plot(pr_curve_df['thresholds'],pr_curve_df['recall'],c=c[index],label=name)
	#ax3_p.plot(pr_curve_df['recall'],pr_curve_df['precision'],c=c[index],label=name)

#c=list(mcolors.TABLEAU_COLORS)
c=np.append(list(mcolors.TABLEAU_COLORS),list(mcolors.BASE_COLORS))

#save_path = path_save_eval
#ml_class = ['custom', 'linear']
#ml_class = ['custom']

#fontsize = 20
'''
for fuzzy_option in fuzzy_options:
    fig, ((ax1_p, ax2_p, ax3_p), (ax1_r, ax2_r, ax3_r)) = plt.subplots(2,3)		
    fig.suptitle(f'PR_curve and ROC_curve general data')
    ax1_p.set_xlabel('Thresholds',fontsize=fontsize)
    ax1_p.set_ylabel('Precision',fontsize=fontsize)
    ax2_p.set_xlabel('Thresholds',fontsize=fontsize)
    ax2_p.set_ylabel('Recall',fontsize=fontsize)
    ax3_p.set_xlabel('Recall',fontsize=fontsize)
    ax3_p.set_ylabel('Precision',fontsize=fontsize)
    ax1_r.set_xlabel('Thresholds',fontsize=fontsize)
    ax1_r.set_ylabel('FPR',fontsize=fontsize)
    ax2_r.set_xlabel('Thresholds',fontsize=fontsize)
    ax2_r.set_ylabel('TPR',fontsize=fontsize)
    ax3_r.set_xlabel('FPR',fontsize=fontsize)
    ax3_r.set_ylabel('TPR',fontsize=fontsize)
    fig.set_size_inches(25,20)
    index=0
    ###
    for name_s in name_sample:
    	for cl in name_classifire:
            nam_clas = "AGN"
            if(name_s == "sfg"):
                nam_clas = "SFG"
            elif(name_s == "qso"):
                nam_clas = "QSO"

            data_general = pd.read_csv(f"{path_classifire}_{name_s}/{cl}/{fuzzy_option}/{fuzzy_option}_generalization.csv",sep=",",header=0)
            #print(data_general)
            label = []
            n=data_general.shape[0]
            print(n)

            for i in range(n):
                if (data_general['name'].iloc[i] == nam_clas):
                    label.append(1)
                else: label.append(0)
            #print(label)
            
            ev = eval(label,data_general['y_pred'],n)
            ev.to_csv(f'{path_save_eval}/{name_s}_{cl}_{fuzzy_option}_evaluate.csv', index=False)
            
            i_need_more_eval(ax1_p, ax2_p, ax3_p, ax1_r, ax2_r, ax3_r, index, f"{name_s}_{cl}_{fuzzy_option}", label, data_general['y_prob_positive_class'])
            index+=1
    ###
    for name_s in name_sample:
        for ml_c in ml_class:
            data_NN = pd.read_csv(f"{path_save_eval}/{name_s}_{ml_c}_{1}_prob.csv",sep=",",header=0)
            n=data_NN.shape[0]
            label = []
            for i in range(n):
                if(data_NN['y_prob'].iloc[i] > 0.5):
                    label.append(1)
                else:
                    label.append(0)
            ev = eval(data_NN['Y'], label, n)
            ev.to_csv(f"{path_save_eval}/{name_s}_{ml_c}_{1}_evaluate.csv", index=False)
            i_need_more_eval(ax1_p, ax2_p, ax3_p, ax1_r, ax2_r, ax3_r, index, f"{name_s}_{ml_c}_{1}", data_NN['Y'], data_NN['y_prob'])
            index+=1
    ax1_p.legend(loc=3, prop={'size': legend_size})
    ax2_p.legend(loc=3, prop={'size': legend_size})
    ax3_p.legend(loc=3, prop={'size': legend_size})
    ax1_r.legend(loc=1, prop={'size': legend_size})
    ax2_r.legend(loc=3, prop={'size': legend_size})
    ax3_r.legend(loc=4, prop={'size': legend_size})
    
    #fig.savefig(save_path+'/'+fuzzy_option+'PR_ROC_curve_all_g_test_sfg_a.png')	
    fig.savefig(save_path+'/'+'PR_ROC_curve_all_g_test___.png')	
    plt.close(fig)
'''
#classif = ['et_not_b_fuzzy_dist', 'rf_not_b_fuzzy_dist', 'normal_1']
#classif = ['custom_1']

#cls_name = ['extremely randomized tree', 'random forest', 'neural network']
#cls_name = ['neural network']


#for name_s in name_sample:
def ROC_picture(path_save_eval,name,classif):
	fontsize_sub = 50
	fontsize_label = 24
	fontsizr_param = 24
	fontsize_legend = 20
	size = (12,11)

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(1,1,1)
	#fig1.suptitle("", fontsize=fontsize_sub)
	ax1.set_xlabel("Probability thresholds", fontsize=fontsize_label)
	ax1.set_ylabel("Precision", fontsize=fontsize_label)
	ax1.tick_params(axis='x', labelsize=fontsizr_param)
	ax1.tick_params(axis='y', labelsize=fontsizr_param)
	fig1.set_size_inches(size)

	fig2 = plt.figure()
	ax2 = fig2.add_subplot(1,1,1)
	#fig2.suptitle("", fontsize=fontsize_sub)
	ax2.set_xlabel("Probability thresholds", fontsize=fontsize_label)
	#ax2.set_ylabel("Completeness", fontsize=fontsize_label)
	ax2.set_ylabel("Recall", fontsize=fontsize_label)
	ax2.tick_params(axis='x', labelsize=fontsizr_param)
	ax2.tick_params(axis='y', labelsize=fontsizr_param)
	fig2.set_size_inches(size)

	fig3 = plt.figure()
	ax3 = fig3.add_subplot(1,1,1)
	#fig3.suptitle("", fontsize=fontsize_sub)
	#ax3.set_xlabel("Completeness", fontsize=fontsize_label)
	ax3.set_xlabel("Recall", fontsize=fontsize_label)
	ax3.set_ylabel("Precision", fontsize=fontsize_label)
	ax3.tick_params(axis='x', labelsize=fontsizr_param)
	ax3.tick_params(axis='y', labelsize=fontsizr_param)
	fig3.set_size_inches(size)

	fig4 = plt.figure()
	ax4 = fig4.add_subplot(1,1,1)
	#fig4.suptitle("", fontsize=fontsize_sub)
	ax4.set_xlabel("FPR", fontsize=fontsize_label)
	ax4.set_ylabel("TPR", fontsize=fontsize_label)
	ax4.tick_params(axis='x', labelsize=fontsizr_param)
	ax4.tick_params(axis='y', labelsize=fontsizr_param)
	fig4.set_size_inches(size)
	
	index=1
	for cl in classif:
		data_pr = pd.read_csv(f"{path_save_eval}/{name}_{cl}_pr.csv", sep=",", header=0)
		#ax1.plot(data_pr['thresholds'],data_pr['precision'],c=c[index],label=cls_name[index-1])
		ax1.plot(data_pr['thresholds'],data_pr['precision'],c=c[index],label=cl)
		index+=1

	index=1
	for cl in classif:
		data_pr = pd.read_csv(f"{path_save_eval}/{name}_{cl}_pr.csv", sep=",", header=0)		
		ax2.plot(data_pr['thresholds'],data_pr['recall'],c=c[index],label=cl)
		index+=1

	index=1
	for cl in classif:
		data_pr = pd.read_csv(f"{path_save_eval}/{name}_{cl}_pr.csv", sep=",", header=0)
		ax3.plot(data_pr['precision'],data_pr['recall'],c=c[index],label=cl)
		index+=1

	index=1
	for cl in classif:
		data_roc = pd.read_csv(f"{path_save_eval}/{name}_{cl}_roc.csv", sep=",", header=0)
		ax4.plot(data_roc['fpr'],data_roc['tpr'],c=c[index],label=cl)
		index+=1



	ax1.legend(loc=4,prop={'size': fontsize_legend})
	ax2.legend(loc=3,prop={'size': fontsize_legend})
	ax3.legend(loc=3,prop={'size': fontsize_legend})
	ax4.legend(loc=4,prop={'size': fontsize_legend})
	fig1.savefig(f"{path_save_eval}/{name}_pr_th.png")
	fig2.savefig(f"{path_save_eval}/{name}_rc_th.png")
	fig3.savefig(f"{path_save_eval}/{name}_pr_rc.png")
	fig4.savefig(f"{path_save_eval}/{name}_roc.png")
	plt.close(fig1)
	plt.close(fig2)
	plt.close(fig3)
	plt.close(fig4)

#path_save_eval = '/home/lrikozavr/ML_work/des_pro/ml/eval'
path_save_eval = '/home/lrikozavr/ML_work/allwise_gaiadr3/ml/eval'
pre_name = 'qso_gal_star_w123_wol_full_phot_1021_custom_sm'
#pre_name = 'extragal_custom_sm'
name = '3n64n64n64'
#name = '3n15n11n7'
#
#name = '3n14n10n7'
#
#name = '3n12n9n4'
cls_name_mass = ['star_cls','gal_cls','qso_cls']
data = pd.read_csv(f'{path_save_eval}/{pre_name}_{name}_prob.csv',header=0,sep=',')
for cls_name in cls_name_mass:
	print(cls_name)
	i_need_more_eval(path_save_eval,f'{pre_name}_{name}_{cls_name}',data[cls_name],data[f'{cls_name}_prob'])
	print(eval(data[f'{cls_name}_prob'],data[cls_name],184879))
ROC_picture(path_save_eval,f'{pre_name}_{name}',cls_name_mass)

'''
evalu = pd.DataFrame([np.zeros(14)], columns=['Accuracy','AGN_purity','nonAGN_precision','AGN_completness','nonAGN_completness','F1','FPR','TNR','bACC','K','MCC','BinaryBS','name_classifire','name_object'])
for fuzzy_option in fuzzy_options:
    for name_s in name_sample:
        for ml_c in ml_class:
            data_ev = pd.read_csv(f"{path_save_eval}/{name_s}_{ml_c}_{1}_evaluate.csv",sep=",",header=0)
            print(data_ev)
            data_ev['name_classifire'] = ml_c
            data_ev['name_object'] = name_s
            evalu = evalu.append(data_ev,ignore_index=True)
evalu.to_csv(f"{path_save_eval}/table1.csv",index=False)
'''


