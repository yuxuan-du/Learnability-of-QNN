from tool_box import *
import argparse
import os
r'''
Hyperparameters
'''
parser = argparse.ArgumentParser("QNN-Learnability-Para")
parser.add_argument('--lr', type=float, default=2, help='learning rate')
parser.add_argument('--epoch_num', type=int, default=400, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--batch', type=int, default=280, help='batch size')
parser.add_argument('--num_train_exps', type=int, default=280, help='train dataset size')
parser.add_argument('--num_test_exps', type=int, default=80, help='test dataset size')
parser.add_argument('--n_qubits', type=int, default=3, help='number of qubits')
parser.add_argument('--num_blocks', type=int, default=3, help='number of blocks')
args = parser.parse_args()

r'''
Data load
'''
data_all = np.load('data/binary_mnist_data.npy')
label_all = np.load('data/binary_mnist_label.npy')

num_alpha = args.num_qubit * args.num_blocks
para = np.random.uniform(0, 2*np.pi, num_alpha)
data_train = data_all[0: args.num_train_exps]
data_test = data_all[args.num_train_exps: args.num_train_exps + args.num_test_exps]
lab_train = label_all[0: args.num_train_exps]
lab_test = label_all[args.num_train_exps: args.num_train_exps + args.num_test_exps]

test_acc_list = []
loss_container = []
train_acc_list = []
best_para = np.zeros(num_alpha)
best_vali = 0



print('start to training!')
classfier = PQC_classifier(depth_ = args.num_blocks, num_fea_qubit_= args.num_qubit, num_idx_qubit_=0)
for epoch_ in range(args.epo_number_):
    r''' training procedure '''
    arr = np.arange(args.num_train_exps)
    idx_list = np.random.permutation(arr)
    loss_ = 0
    gradient_ = np.zeros(num_alpha)
    acc_count = 0
    for i in range(args.batch):
        train_idx0 = idx_list[i]
        data_train_tempt = data_train[train_idx0]
        fea_state = feature_encoding(data_train_tempt)
        mea_res = classfier.compute_predict_vector(fea_state, para)
        if mea_res < 0.5 and lab_train[train_idx0] == 0:
            acc_count += 1
        elif mea_res > 0.5 and lab_train[train_idx0] == 1:
            acc_count += 1
        loss_ += loss_func(mea_res, lab_train[train_idx0])
        loss_pos = np.zeros(num_alpha)
        loss_neg = np.zeros(num_alpha)
        r''' Start to optimize the trainable parameters '''
        for j in range(num_alpha):
            para_pos  = para.copy()
            para_pos[j] += np.pi / 2
            loss_pos[j] += classfier.compute_predict_vector(fea_state, para_pos)
            para_neg  = para.copy()
            para_neg[j] -= np.pi / 2
            loss_neg[j] += classfier.compute_predict_vector(fea_state, para_neg)
        gradient_ += 2 * loss_func_grad(mea_res, lab_train[train_idx0]) * (loss_pos - loss_neg) / 2

    gradient = gradient_ / (2 * args.batch)
    para = para - args.learn_rate * gradient
    print('Epoch %d || loss %f || Train acc %f'%(epoch_, loss_/ args.batch, acc_count / args.batch ))
    train_acc_list.append(acc_count / args.num_train_exps)
    loss_container.append(loss_/ args.num_train_exps)

    r''' test procedure '''
    acc_count = 0
    for k in range(args.num_test_exps):
        data_test_tempt = data_test[k]
        fea_state = feature_encoding(data_test_tempt)
        mea_res = classfier.compute_predict_vector(fea_state, para)
        if mea_res < 0.5 and lab_test[k] == 0:
            acc_count += 1
        elif mea_res > 0.5 and lab_test[k] == 1:
            acc_count += 1
    test_acc_list.append( acc_count / args.num_test_exps)
    print('test acc', acc_count / args.num_test_exps)

file_name_para = 'para'+'depth'+str(args.num_blocks)
file_trainloss = 'trainloss'+'depth'+str(args.num_blocks)
file_testacc = 'testacc'+'depth'+str(args.num_blocks)
file_trainacc = 'trainacc'+'depth'+str(args.num_blocks)

np.save(file_name_para, para)
np.save(file_trainacc, train_acc_list)
np.save(file_testacc, test_acc_list)
np.save(file_trainloss, loss_container)

os._exit(1)

