# toxic-comment-classification-nlp-using-pytorch

Perilaku negatif daring, seperti komentar yang tidak senonoh, cenderung membuat orang berhenti mengekspresikan diri dan meninggalkan percakapan. Platform kesulitan mengidentifikasi dan menandai komentar daring yang berpotensi berbahaya atau menyinggung, yang menyebabkan banyak komunitas membatasi atau menutup komentar pengguna sama sekali.

[Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) mengeluarkan tantangan untuk membangun model klasifikasi multilabel yang mampu mendeteksi berbagai jenis toksisitas seperti ancaman, kecabulan, dan hinaan, dan dengan demikian membantu membuat diskusi daring lebih produktif dan penuh rasa hormat.

Data untuk masalah ini adalah kumpulan data dari 159.571 komentar dari suntingan halaman pembicaraan Wikipedia. Komentar-komentar ini telah ditandai sebagai perilaku tidak senonoh oleh pengulas manusia.

# Data Visualization
```
plt.figure(figsize=(15,10))

fr = plt.subplot(2,1,1)
plt.barh(width=vlst,y=klst)
fr.set_xlabel("Total Number of Comments", fontsize=10)
fr.set_ylabel("Number of Classes comments belong to", fontsize=10)
fr.set_title("Plot including unlabeled(0) comments ", fontsize=10)
for key in klst:
    fr.annotate(text=multi_label_total[key], xy=(multi_label_total[key],key), xycoords="data",size=10, va="center")

se = plt.subplot(2,1,2)
plt.barh(width=vlst_0,y=klst_0)
se.set_xlabel("Total Number of Comments", fontsize=10)
se.set_ylabel("Number of Classes comments belong to", fontsize=10)
se.set_title("Plot excluding unlabeled(0) comments", fontsize=10)
for key in klst_0:
    se.annotate(text=multi_label_total[key], xy=(multi_label_total[key],key), xycoords="data",size=10, va="center")
```

![image](https://github.com/user-attachments/assets/4875ca64-031c-4143-999b-4462f84609ac)

```
# plotting distribution of classes 

font1 = {"size":12}
all_tox = list(Total_of_Class.values())
plt.figure(figsize=(20,10))
#plt.rcParams.update({'font.size': 12})
plt.rc("font",**font1)

left = plt.subplot(1,2,1)
plt.bar(x = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate','non_toxic'], height = Total_of_Class.values(), color="#00DFFF")
left.set_facecolor("black")
left.set_title("distribution of classes with non toxic values", fontsize=22)
for key in ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate','non_toxic']:
    left.annotate(text=Total_of_Class[key], xy=(key,Total_of_Class[key]+1000), xycoords="data", color="white", size=14, ha="center")

right = plt.subplot(1,2,2)
plt.bar(x = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate'], height = all_tox[:-1], color="orange")
right.set_facecolor("black")
right.set_title("distribution of classes without non toxic values",fontsize=22)
for key in ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']:
    right.annotate(text=Total_of_Class[key], xy=(key,Total_of_Class[key]+100), xycoords="data", color="white", size=14, ha="center")
```

![image](https://github.com/user-attachments/assets/99322e64-95a1-4dd4-9237-f06e71dad9fd)

# Model building

```
from transformers import DistilBertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Using Pretrained DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# DistilBERT

from transformers import DistilBertForSequenceClassification

Distil_bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

Distil_bert.classifier = nn.Sequential(
                    nn.Linear(768,7),
                    nn.Sigmoid()
                  )
print(Distil_bert)
```

# Evaluation Model

```
def Acc_Loss_Plot(TA1, TA2, TL1, TL2, VA1, VA2, VL1, VL2):
    plt.figure(figsize=(20,14))
    Ac = plt.subplot(2,2,1)
    plt.plot(TA1.keys(), TA1.values())
    plt.plot(TA2.keys(), TA2.values())
    Ac.set_xlabel("Step Number", fontsize=15)
    Ac.set_ylabel("Accuracy", fontsize=15)
    Ac.set_title("Training Accuracy", fontsize=20)
    Ac.legend(["Epoch 1","Epoch 2"])

    Ls = plt.subplot(2,2,2)
    plt.plot(TL1.keys(), TL1.values())
    plt.plot(TL2.keys(), TL2.values())
    Ls.set_xlabel("Step Number", fontsize=15)
    Ls.set_ylabel("Loss", fontsize=15)
    Ls.set_title("Training Loss", fontsize=20)
    Ls.legend(["Epoch 1","Epoch 2"])
    
    VAc = plt.subplot(2,2,3)
    plt.plot(VA1.keys(), VA1.values())
    plt.plot(VA2.keys(), VA2.values())
    VAc.set_xlabel("Step Number", fontsize=15)
    VAc.set_ylabel("Accuracy", fontsize=15)
    VAc.set_title("Validation Accuracy", fontsize=20)
    VAc.legend(["Epoch 1","Epoch 2"])

    VLs = plt.subplot(2,2,4)
    plt.plot(VL1.keys(), VL1.values())
    plt.plot(VL2.keys(), VL2.values())
    VLs.set_xlabel("Step Number", fontsize=15)
    VLs.set_ylabel("Loss", fontsize=15)
    VLs.set_title("Validation Loss", fontsize=20)
    VLs.legend(["Epoch 1","Epoch 2"])

Acc_Loss_Plot(TA[0], TA[1], TL[0], TL[1], VA[0], VA[1], VL[0], VL[1])
```

![image](https://github.com/user-attachments/assets/93d04e73-e2d4-4dd4-a3df-40951cf5163b)

#Save Model
```
# Saving model
torch.save(Distil_bert,"dsbert_toxic_balanced.pt")
```




