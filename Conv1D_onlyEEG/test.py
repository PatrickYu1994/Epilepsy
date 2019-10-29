import scipy.io as scio
from train import xs_gen
from keras.models import load_model
from utils import auc, f1
from sklearn import metrics

test_set_path = "./data_set/test_set.mat"
best_model = './result/best_model/20190804best_model.215-0.6518.h5'

test_set = xs_gen(scio.loadmat(test_set_path))
model = load_model(best_model, custom_objects={"auc": auc, "f1": f1})
prediction = model.predict_classes(test_set[0])
y_true = scio.loadmat(test_set_path)['y']
y_true = [arr[1] for arr in y_true]
fpr, tpr, thresholds = metrics.roc_curve(y_true, prediction)

val_auc = metrics.auc(fpr, tpr)
f_score = metrics.f1_score(y_true=y_true, y_pred=prediction)
print("auc: ", val_auc)
print("f_sc: ", f_score)


