import scipy.io as scio
from main import xs_gen
from keras.models import load_model
from utils import auc, f1
from sklearn import metrics
import numpy as np
#np.set_printoptions(threshold=np.inf)

test_set = xs_gen(scio.loadmat("./gen_dataset/test_set.mat"))

model = load_model('./Model/v1_2019_07_17_pnrate_1_2.h5', custom_objects={"auc": auc, "f1": f1})

#prediction = model.predict(test_set[0])
#p_rate = prediction[:, 1]
prediction = model.predict_classes(test_set[0])
y_true = scio.loadmat("./gen_dataset/test_set.mat")['y']
y_true = [arr[0] for arr in y_true]

#fpr, tpr, thresholds = metrics.roc_curve(y_true, p_rate)
#fpr, tpr, thresholds = metrics.roc_curve(y_true, prediction)
#val_auc = metrics.auc(fpr, tpr) # 1_5, v2: 0.5732863286555002
                                # 1_2, v1: 0.7396738206950123
#print(val_auc)

acc_score = metrics.accuracy_score(y_true, prediction) # 1_5, v2: 0.8200354609929078
                                                       # 1_2, v1: 0.7088764742396028
print(acc_score)

