import scipy.io as scio
from main import xs_gen
from keras.models import load_model
from utils import auc, f1
from sklearn import metrics
import numpy as np
#np.set_printoptions(threshold=np.inf)

test_set = xs_gen(scio.loadmat("./gen_dataset/test_set.mat"))

model = load_model('./Model/20190723best_model.50-0.7061.h5', custom_objects={"auc": auc, "f1": f1})

#prediction = model.predict(test_set[0])
#p_rate = prediction[:, 1]
prediction = model.predict_classes(test_set[0])
y_true = scio.loadmat("./gen_dataset/test_set.mat")['y']
y_true = [arr[0] for arr in y_true]

#fpr, tpr, thresholds = metrics.roc_curve(y_true, p_rate)
fpr, tpr, thresholds = metrics.roc_curve(y_true, prediction)
val_auc = metrics.auc(fpr, tpr) # 1_5, v2: auc:0.5732863286555002
                                # 1_2, v1: auc:0.7396738206950123
                                # 1_2, CNN3D V1: auc:0.4302690582959642 / fscore:0.1211890243902439
f_score = metrics.f1_score(y_true=y_true, y_pred=prediction)
print("auc: ", val_auc)
print("f_sc: ", f_score)

