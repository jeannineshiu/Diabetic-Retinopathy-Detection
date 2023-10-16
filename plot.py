import sys
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pprint import pprint

def show_result(res_list):
    fig, ax = plt.subplots()
    for res in res_list:
        for key,val in res[1].items():
            ax.plot(res[0], val, label=key)

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy(%)")
    ax.legend()
    ax.grid()
    # ax.set_title("ResNet comparision(ResNet18)")
    ax.set_title("ResNet comparision(ResNet50)")
    plt.show()

def show_confusion_matrix(y, gt, cl):
    cm = confusion_matrix(gt, y)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=cl, yticklabels=cl,
           title='resnet50_pretrained',
           ylabel='True label',
           xlabel='Predicted label')
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "white")
    fig.tight_layout()
    plt.show()

def main():
    source = {}
    # files = ['resnet18_pretrained_sgd_ep10_b4_lr0.001_wd0.0005.json','resnet18_untrained_sgd_ep10_b4_lr0.001_wd0.0005.json']
    files = ['resnet50_pretrained_sgd_ep10_b4_lr0.001_wd0.0005.json','resnet50_untrained_sgd_ep10_b4_lr0.001_wd0.0005.json']
    # files = ['resnet50_pretrained_sgd_ep5_b4_lr0.001_wd0.0005.json']
    result_list = []
    for fi in files:
        with open(fi, 'r') as f:
            source = json.load(f)
        print(source['title'])
        pprint(list(zip(source['y_dict'].keys(), [max(i) for i in source['y_dict'].values()])))
        result_list.append([source['x'], source['y_dict'], source['title']])
    show_result(result_list)

    source = {}
    #cm_file = ['resnet18_untrained.json']
    #cm_file = ['resnet18_untrained.json']
    #cm_file = ['resnet50_untrained.json']
    cm_file = ['resnet50_pretrained.json']
    for fi in cm_file:
        with open(fi, 'r') as f:
            source = json.load(f)
            show_confusion_matrix(source['pred_y'], source['gt'], source['class'])
    

if __name__ == "__main__":
    main()
    
#%%