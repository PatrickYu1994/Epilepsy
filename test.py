from sklearn.metrics import auc
import scipy.io as scio
from main import xs_gen
from keras.models import load_model
from utils import auc, f1


test_set = xs_gen(scio.loadmat("./gen_dataset/test_set.mat"))

model = load_model('./Model/v1_2019_07_17_pnrate_1_2.h5', custom_objects={"auc": auc, "f1": f1})

prediction = model.predict(test_set[0])
true_y = test_set[1]

print(prediction)

