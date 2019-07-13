import scipy.io as scio

training_set = scio.loadmat("./test_train.mat")
validation_set = scio.loadmat("./test_val.mat")
test_set = scio.loadmat("./test_test.mat")
