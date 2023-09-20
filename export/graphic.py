  # -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

#from main import save_path
from network import LoadModel, make_custom_index, dimention_reduction_tsne, get_features


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#from main import dir
def dir(save_path,name):
  dir_name = f"{save_path}/{name}"
  if not os.path.isdir(dir_name):
      os.mkdir(dir_name)

def MCD_plot(name,d):
    count = len(d)
    ar_i = [i for i in range(count)]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(10,10)
    ax.scatter(ar_i, d, color = colors[0])
    fig.savefig(f"{save_path}/{name}_MCD_distance.png")
    plt.close(fig)    

def TSNE_pic(config):
    
    data_b = pd.read_csv(f"{config.path_ml_data}/{config.name_main_sample}_all.csv", header=0, sep=',')
    
    data_b = data_b.sample(100000, ignore_index=True)

    data = data_b[get_features(config.features["train"],config)]

    label = data_b[config.name_class_cls]
    
    del data_b

    data = dimention_reduction_tsne(data.values,config)
    
    data = pd.DataFrame(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    for n, name in enumerate(config.name_class_cls):
        temp_data = data[label[name] == 1].values
        print(temp_data)
        ax.scatter(temp_data[:,0],temp_data[:,1], color = colors[n], label=name)
        del temp_data
    
    data.to_csv('',index=False)

    del data, label

    fig.legend()
    fig.set_size_inches(10,10)

    fig.savefig(f"{config.path_pic}/{config.name_sample}_tsne.png")
    plt.close(fig)


def contam_dist_pic(config):
    
    def contamination_distribution(data,features_name,config):
        bins = config.picture["contam_dist"]["bins"]
        min = data[features_name].min()
        max = data[features_name].max()
        #range_bins = max-min
        mass_mags = np.linspace(min, max, num = bins)
        cls_n = len(config.name_class)
        mass = np.zeros((bins,cls_n*cls_n))
        contam_mass = np.zeros((bins,cls_n*cls_n))

        sum_global = np.zeros((bins,cls_n*cls_n))
        for i in range(bins):
            #min_f = (range_bins*i)/bins + min
            #max_f = (range_bins*(i+1))/bins + min
            if(i+1 == bins):
                break
            min_f = mass_mags[i]
            max_f = mass_mags[i+1]
            #потребує оптимізації, бо O(n^2)
            data_temp = data[(data[features_name] >= min_f) & (data[features_name] < max_f)]
            #print(data_temp)
            #
            y = np.argmax(data_temp[config.name_class_cls], axis=1).tolist()
            y_prob = np.argmax(data_temp[config.name_class_prob], axis=1).tolist()
            #        
            cm = confusion_matrix(y, y_prob)
            if(len(cm)>2):
              #print(cm)
              #print(y,y_prob)
              for ii in range(cls_n):
                  sum = 0
                  for jj in range(cls_n):
                      #all correct
                      mass[i][cls_n*ii+jj] = cm[ii][jj]
                      sum += cm[ii][jj]
                      if(i-1 > 0):
                         sum_global[i][cls_n*ii+jj] += sum_global[i-1][cls_n*ii+jj]
                      sum_global[i][cls_n*ii+jj] += cm[ii][jj] 
                  for jj in range(cls_n):
                      if(sum!=0):
                          contam_mass[i][cls_n*ii+jj] = (cm[ii][jj]/float(sum))*100

        '''            
        fig, axs = plt.subplots(cls_n,cls_n)
        
        for n in range(cls_n*cls_n):
            ii, jj = n//cls_n, n%cls_n
            axs[ii,jj].plot(mass_mags,mass[:,n])
            axs[ii,jj].set_xlabel(features_name)
            axs[ii,jj].set_ylabel('Contamination count')
            axs[ii,jj].set_xlim([min-1,max+1])
        '''
        fig, axs = plt.subplots(3,cls_n)

        for n in range(cls_n):

            sum_local = 0
            for ii in range(cls_n):
                sum_local += sum_global[bins-2,cls_n*n+ii]

            for ii in range(cls_n):
                if(ii != n):
                    axs[1,n].plot(mass_mags,contam_mass[:,cls_n*n+ii],label=f"{config.name_class_cls[n]} as {config.name_class_prob[ii]}")
                    axs[0,n].plot(mass_mags,np.log10(mass[:,cls_n*n+ii]),label=f"{config.name_class_cls[n]} as {config.name_class_prob[ii]}")
                    axs[2,n].plot(mass_mags,(sum_global[:,cls_n*n+ii]/sum_local)*100,label=f"{config.name_class_cls[n]} as {config.name_class_prob[ii]}")
                else:
                    axs[0,n].plot(mass_mags,np.log10(mass[:,cls_n*n+ii]),label=f"{config.name_class_cls[n]} true pred")
            axs[1,n].set_xlabel("mags")
            axs[1,n].set_ylabel('Contamination, % (per bin)')

            axs[0,n].set_xlabel("mags")
            axs[0,n].set_ylabel('Contamination count, log10 (per bin)')
            
            axs[2,n].set_xlabel("mags")
            axs[2,n].set_ylabel('Contamination, %')

            axs[2,n].set_xlim([min-1,max+1])
            axs[0,n].set_xlim([min-1,max+1])
            axs[1,n].set_xlim([min-1,max+1])

            axs[2,n].legend()
            axs[1,n].legend()
            axs[0,n].legend()
            #axs[1,n].set_ylim([,])
        
        #fig.legend()
        fig.supxlabel(features_name, fontsize=30)
        fig.set_size_inches(30,30)

        fig.savefig(f'{config.path_pic}/{config.name_sample}_cm_contamination_by_{features_name}.png')
        plt.close(fig)


    name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
    label = pd.read_csv(f'{config.path_eval}_custom_sm_{name}_prob.csv', header=0, sep=",")
    
    data = pd.read_csv(f"{config.path_ml_data}/{config.name_main_sample}_all.csv", header=0, sep=',')

    for name in get_features(["mags"],config):
        data_temp = data[name]
        data_temp = pd.concat((data_temp,label),axis=1)
        contamination_distribution(data_temp,name,config)


def multigridplot(data, features, config):
    count = len(features)
    fig, axs = plt.subplots(count,count)

    fig.set_size_inches(30,30)

    #add input features weights

    for index, name_index in enumerate(config.name_class_cls):
        data_class = data[data[name_index] == 1]
        for ii, name_ii in enumerate(features):
            for jj, name_jj in enumerate(features):
                if(not ii==jj and ii > jj):
                    data_common = pd.concat([data_class[name_ii],data_class[name_jj]], axis=1)
                    sns.kdeplot(data=data_common, ax=axs[ii,jj], color=colors[index])
                    axs[ii,jj].scatter(data_class[name_jj],data_class[name_ii], color = colors[index])
                else:
                    axs[ii,jj].remove()
    
    fig.legend()

    fig.savefig(f'{config.path_pic}/{config.name_sample}_multyhist_distribution.png')
    plt.close(fig)

def picture_cm(config):
  def plot_cm(index,save_name):
    name = make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"])
    data = pd.read_csv(f'{config.path_eval}_custom_sm_{name}_prob.csv', header=0, sep=",")
    
    y = np.argmax(data[config.name_class_cls], axis=1).tolist()
    y_prob = np.argmax(data[config.name_class_prob], axis=1).tolist()

    cm = confusion_matrix(y, y_prob)
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.savefig(f'{config.path_pic}/{save_name}_Confusion_matrix.png')
    plt.close(fig)

  for i in range(config.hyperparam["model_variable"]["kfold"]):
    plot_cm(i,f"{config.name_sample}_{i}_kfold")
  #
  if(config.picture["main"]["work"]):
    plot_cm('00',f"{config.name_sample}_main")
  #
  print("picture Confusion Matrix done")

def picture_metrics(config):
    def plot_metrics(axs, name, history):
      metrics = ['loss', 'auc', 'precision', 'recall']
      #fig, axs = plt.subplots(2,2)

      for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        i, j = n//2, n%2
        axs[i,j].plot(history.epoch, history.history[metric], color=colors[0], label='Train_' + name)
        axs[i,j].plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val_' + name)
        axs[i,j].xlabel('Epoch')
        axs[i,j].ylabel(name)
        if metric == 'loss':
          axs[i,j].ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          axs[i,j].ylim([0.8,1])
        else:
          axs[i,j].ylim([0,1])

        #axs[i,j].legend()

    fig, axs = plt.subplots(2,2)

    for i in range(config.hyperparam["model_variable"]["kfold"]):
        name = make_custom_index(i,config.hyperparam["model_variable"]["neuron_count"])
        model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",config.hyperparam["optimizer"],config.hyperparam["loss"])
        plot_metrics(axs, f"{i}_kfold", model)

    if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
        name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
        model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",config.hyperparam["optimizer"],config.hyperparam["loss"])
        plot_metrics(axs, f"main", model)

    fig.legend()
    fig.set_size_inches(30,20)
    if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
        fig.savefig(f'{config.path_pic}/{config.name_sample}_kfold_main_summary_metrics_history.png')
    else:
        fig.savefig(f'{config.path_pic}/{config.name_sample}_kfold_summary_metrics_history.png')
       
    plt.close(fig)

    if(config.picture["main"]["work"] and not config.picture["main"]["bound"]):
        fig, axs = plt.subplots(2,2)
        name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
        model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",optimizer,loss)
        plot_metrics(axs, f"main", model)
        fig.legend()
        fig.set_size_inches(30,20)
        fig.savefig(f'{config.path_pic}/{config.name_sample}_main_metrics_history.png')
        plt.close(fig)        

def picture_hist(config):
    
    def Hist1(ax,x,mag,label, **kwargs):
        ax.set_xlabel(mag,fontsize=40)
        ax.set_ylabel("count",fontsize=40)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        #ax.set_title(name,fontsize = 50)
        ax.hist(x,bins=200, label=label,**kwargs)
  
    dir(config.path_pic,'hist')

    data_mass = []
    for class_name in config.name_class:
        data_temp = pd.read_csv(f"{config.path_ml_data}/{config.name_main_sample}_{class_name}_main_sample.csv", header=0, sep=',')       
        data_mass.append(data_temp)

    columns = data_mass[0].drop(config.base, axis=1).columns.values

    if(not config.picture["hist"]["bound"]):
        for col in columns:
            for i, class_name in enumerate(config.name_class):
                fig=plt.figure()
                ax = fig.add_subplot(1,1,1)
                fig.suptitle(col, fontsize=50)       
                Hist1(ax,data_mass[i][col],col,class_name, histtype='step', fill=config.picture["hist"]["fill"])
                ax.legend(prop={'size': 30})
                fig.set_size_inches(30,20)
                fig.savefig(f"{config.path_pic}/hist/{config.name_sample}_{class_name}_{col}_hist.png")
                plt.close(fig)
    elif(config.picture["hist"]["bound"]):
        for col in columns:
            fig=plt.figure()
            ax = fig.add_subplot(1,1,1)
            fig.suptitle(col, fontsize=50)       
            hist_data = []
            for i, class_name in enumerate(config.name_class):
                hist_data.append(data_mass[i][col])
            Hist1(ax,hist_data,col,config.name_class, histtype='step', stacked=config.picture["hist"]["stacked"], fill=config.picture["hist"]["fill"])
            ax.legend(prop={'size': 30})
            fig.set_size_inches(30,20)
            fig.savefig(f"{config.path_pic}/hist/{config.name_sample}_{col}_hist.png")
            plt.close(fig)
    else:
       raise Exception('unknown config value config.picture["hist"]["bound"] ', config.picture["hist"]["bound"])

    print("picture Hist done")

def picture_loss(optimizer,loss,config):

  def plot_loss(ax, history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    ax.semilogy(history.epoch, history.history['loss'],
                color=colors[n], label='Train ' + label)
    ax.semilogy(history.epoch, history.history['val_loss'],
                color=colors[n], label='Val ' + label,
                linestyle="--")
    ax.xlabel('Epoch')
    ax.ylabel('Loss')

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  
  for i in range(config.hyperparam["model_variable"]["kfold"]):
    name = make_custom_index(i,config.hyperparam["model_variable"]["neuron_count"])
    model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",optimizer,loss)
    model.fit()
    plot_loss(ax,model,f"kfold_{i}",i)
  #
  if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
    name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
    model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",optimizer,loss)
    plot_loss(ax,model,f"main",config.hyperparam["model_variable"]["kfold"] + 1)
  #
  ax.set_label(config.name_sample)
  ax.legend()
  if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
    fig.savefig(f'{config.path_pic}/{config.name_sample}_kfold_main_summary_loss.png')     
  else:
    fig.savefig(f'{config.path_pic}/{config.name_sample}_kfold_summary_loss.png')

  #for main
  if(config.picture["main"]["work"] and not config.picture["main"]["bound"]):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
    model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weights}_custom_sm_{name}",optimizer,loss)
    plot_loss(ax,model,f"main",config.hyperparam["model_variable"]["kfold"] + 1)

    ax.set_label(config.name_sample)
    ax.legend()
    fig.savefig(f'{config.path_pic}/{config.name_sample}_main_loss.png')

  del fig

  

def picture_roc_prc(config):  

  def plot_roc(ax, name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    #fig = plt.figure(figsize=(5,5))
    ax.plot(fp, tp, label=name, linewidth=2, **kwargs)
    ax.set_xlabel('False positives [%]', fontsize=24)
    ax.set_ylabel('True positives [%]', fontsize=24)
    ax.set_xlim(config.picture["roc_prc"]["lim_roc"][0])
    ax.set_ylim(config.picture["roc_prc"]["lim_roc"][1])
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(True)
    ax = plt.gca()
    #ax.legend()
    #ax.set_aspect('equal')
    #fig.savefig(f'{save_path}/{name}_ROC.png')
    #plt.close(fig)

  def plot_prc(ax, name, labels, predictions, **kwargs):
      precision, recall, _ = precision_recall_curve(labels, predictions)

      ax.plot(precision, recall, label=name, linewidth=2, **kwargs)
      ax.set_xlim(config.picture["roc_prc"]["lim_prc"][0])
      ax.set_ylim(config.picture["roc_prc"]["lim_prc"][1])
      ax.set_xlabel('Precision', fontsize=24)
      ax.set_ylabel('Recall', fontsize=24)
      ax.tick_params(axis='x', labelsize=30)
      ax.tick_params(axis='y', labelsize=30)
      ax.grid(True)
      ax = plt.gca()
      #ax.set_aspect('equal')
      #ax.legend()
  
  dir(config.path_pic,'roc_prc')

  name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
  data_main = pd.read_csv(f'{config.path_eval}_custom_sm_{name}_prob.csv', header=0, sep=",")

  data_mass = []
  for i in range(config.hyperparam["model_variable"]["kfold"]):
    name = make_custom_index(i,config.hyperparam["model_variable"]["neuron_count"])
    data = pd.read_csv(f'{config.path_eval}_custom_sm_{name}_prob.csv', header=0, sep=",")
    data_mass.append(data)
  
  for key in config.picture["roc_prc"]["flags"]:
    match key:
        case 1:
            for class_name in config.name_class:
                fig = plt.figure()
                ax_prc = fig.add_subplot(1,2,2)
                ax_roc = fig.add_subplot(1,2,1)
                for i in range(config.hyperparam["model_variable"]["kfold"]):
                    plot_prc(ax_prc,f"{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
                    plot_prc(ax_prc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])                  
                ax_roc.legend(prop={'size': 15})
                ax_prc.legend(prop={'size': 15})
                fig.set_label(f"{config.name_sample}_{class_name}")
                #fig.legend()
                fig.set_size_inches(30,15)
                fig.savefig(f'{config.path_pic}/roc_prc/{config.name_sample}_{class_name}_kfold_summary_roc_prc.png')
                plt.close(fig)
        case 2:
            fig = plt.figure()
            ax_prc = fig.add_subplot(1,2,2)
            ax_roc = fig.add_subplot(1,2,1)
            for class_name in config.name_class:
                for i in range(config.hyperparam["model_variable"]["kfold"]):
                    plot_prc(ax_prc,f"{i}_{class_name}",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"{i}_{class_name}",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
                    plot_prc(ax_prc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])                  
            ax_roc.legend(prop={'size': 15})
            ax_prc.legend(prop={'size': 15})
            fig.set_size_inches(30,15)          
            fig.set_label(f"{config.name_sample}")
            #fig.legend()
            fig.savefig(f'{config.path_pic}/roc_prc/{config.name_sample}_class_kfold_summary_roc_prc.png')
            plt.close(fig)
        case 3:
            for i in range(config.hyperparam["model_variable"]["kfold"]):
                fig = plt.figure()
                ax_prc = fig.add_subplot(1,2,2)
                ax_roc = fig.add_subplot(1,2,1)
                for class_name in config.name_class:
                    plot_prc(ax_prc,class_name,data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,class_name,data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
                        plot_prc(ax_prc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])
                        plot_roc(ax_roc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])                  
                ax_roc.legend(prop={'size': 15})
                ax_prc.legend(prop={'size': 15})
                fig.set_size_inches(30,15)          
                fig.set_label(f"{config.name_sample}_{i}_kfold")
                #fig.legend()
                fig.savefig(f'{config.path_pic}/roc_prc/{config.name_sample}_{i}_kfold_class_summary_roc_prc.png')
                plt.close(fig)
        case 4:
            for i in range(config.hyperparam["model_variable"]["kfold"]):
                for class_name in config.name_class:
                    fig = plt.figure()
                    ax_prc = fig.add_subplot(1,2,2)
                    ax_roc = fig.add_subplot(1,2,1)
                    plot_prc(ax_prc,f"{class_name}_{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"{class_name}_{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    if(config.picture["main"]["work"] and config.picture["main"]["bound"]):
                        plot_prc(ax_prc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])
                        plot_roc(ax_roc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])                  
                    ax_roc.legend(prop={'size': 15})
                    ax_prc.legend(prop={'size': 15})
                    fig.set_size_inches(30,15)
                    fig.set_label(f"{config.name_sample}_{class_name}_{i}_kfold")
                    #fig.legend()
                    fig.savefig(f'{config.path_pic}/roc_prc/{config.name_sample}_{i}_kfold_{class_name}_summary_roc_prc.png')
                    plt.close(fig)
  
  if(config.picture["main"]["work"] and not config.picture["main"]["bound"]):
      if(4, 1 in config.picture["roc_prc"]):
          for class_name in config.name_class:
              fig = plt.figure()
              ax_prc = fig.add_subplot(1,2,2)
              ax_roc = fig.add_subplot(1,2,1)
              plot_prc(ax_prc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])
              plot_roc(ax_roc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])                  
              fig.set_label(f"{config.name_sample}_{class_name}_main")
              ax_roc.legend(prop={'size': 15})
              ax_prc.legend(prop={'size': 15})
              fig.set_size_inches(30,15)          
              #fig.legend()
              fig.savefig(f'{config.path_pic}/roc_prc/{config.name_sample}_{class_name}_main_roc_prc.png')
              plt.close(fig)
      elif(2, 3 in config.picture["roc_prc"]):
          fig = plt.figure()
          ax_prc = fig.add_subplot(1,2,2)
          ax_roc = fig.add_subplot(1,2,1)
          for class_name in config.name_class:
              plot_prc(ax_prc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])
              plot_roc(ax_roc,f"main",data_main[f'{class_name}_cls'],data_main[f'{class_name}_cls_prob'])                  
          ax_roc.legend(prop={'size': 15})
          ax_prc.legend(prop={'size': 15})
          fig.set_size_inches(30,15)                    
          fig.set_label(f"{config.name_sample}_main")
          #fig.legend()
          fig.savefig(f'{config.path_pic}/roc_prc/{config.name_sample}_main_summary_roc_prc.png')
          plt.close(fig)
         
  print("picture ROC&PRC done")