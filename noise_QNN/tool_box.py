import numpy as np

def I():
    tempt = np.array([[1., 0],
                      [0, 1.]])
    return  tempt

def H():
    tempt = np.array([[1., 1.],
                      [1., -1.]])/np.sqrt(2)
    return tempt

def X():
    tempt = np.array([[0, 1.],
                      [1., 0]])
    return tempt

def CNOT():
    tempt = np.zeros(shape=(4,4))
    tempt[0, 0], tempt[1, 1] = 1, 1
    tempt[2, 3], tempt[3, 2] = 1, 1
    return tempt

def CZ():
    tempt = np.zeros(shape=(4,4))
    tempt[3, 3] = -1
    return tempt

def RX(theta_):
    tempt = np.array([[np.cos(theta_/2), -1j*np.sin(theta_/2) ],
                      [-1j*np.sin(theta_/2), np.cos(theta_/2)] ])
    return tempt

def RY(theta_):
    tempt = np.array([[np.cos(theta_ / 2), -np.sin(theta_ / 2)],
                      [np.sin(theta_ / 2), np.cos(theta_ / 2)]])
    return tempt

def RZ(theta_):
    tempt = np.array([[np.cos(theta_/2) - 1j* np.sin(theta_/2), 0],
                      [0, np.cos(theta_/2) + 1j* np.cos(theta_/2)]])
    return tempt

def CRY3(theta1_, theta2_, theta3_ ):
    theta_ = (np.pi - theta1_)*(np.pi - theta2_)*(np.pi - theta3_)
    basis0 = np.zeros(shape=(2,2))
    basis0[0,0] = 1
    basis1 = np.zeros(shape=(2,2))
    basis1[1, 1] = 1
    tempt = RY(theta_)
    out1 = np.kron(basis0, I()) + np.kron(basis1, tempt)
    out2 = np.kron(np.kron(basis0, I()), I()) + np.kron(np.kron(basis1, I()), tempt)
    return out1, out2

def CNOT_layer(total_n_qubits):
    if total_n_qubits < 3:
        print('too fewer qubits to run CNOT layer')
        exit(1)
    U_tempt_1 = CNOT()
    U_tempt_2 = I()
    if total_n_qubits % 2 == 0:
        depth = int(total_n_qubits / 2)
        for iter_ in range(depth - 1):
            U_tempt_1 = np.kron(CNOT(), U_tempt_1)
            U_tempt_2 = np.kron(CNOT(), U_tempt_2)
        U_tempt_2 = np.kron(I(), U_tempt_2)
    else:
        depth = int(np.floor(total_n_qubits / 2))
        for iter_ in range(depth - 1):
            U_tempt_1 = np.kron(CNOT(), U_tempt_1)
            U_tempt_2 = np.kron(CNOT(), U_tempt_2)
        U_tempt_1 = np.kron(I(), U_tempt_1)
        U_tempt_2 = np.kron(CNOT(), U_tempt_2)

    return U_tempt_2 @ U_tempt_1

def feature_encoding(feature_, n_qubit =3):
    r'''
    :param feature_: The reduced dimensions of features
    :return: Generated state
    '''
    Hada = np.kron(H(), np.kron(H(),H()) )
    U_layer1 = np.kron(RY(feature_[2]), np.kron(RY(feature_[1]), RY(feature_[0])))
    CRY2_, CRY_I_2 = CRY3(feature_[2],  feature_[1], feature_[0])
    U_layer2 = np.kron(CRY2_, I()) @ np.kron(I(), CRY2_) #@ CRY_I_2
    U_all =   U_layer2 @ U_layer1 @ Hada @ U_layer2 @ U_layer1 @ Hada  @ U_layer2 @ U_layer1 @ Hada
    Init_state = np.zeros(2**n_qubit)
    Init_state[0] = 1
    out = U_all @ Init_state
    return out

class PQC_classifier():
    def __init__(self, depth_, num_fea_qubit_, num_idx_qubit_):
        r'''
        :param depth_: circuit depth
        :param para_:  input trainable parameters
        :param num_fea_qubit_: number of qubits for feature register
        :param num_idx_qubit_: number of qubits for index register, index refers to the maximum length of token
        :param the target dimension, e.g., k classification has target_dim_ = k
        '''
        self.depth_ = depth_
        self.num_fea_qubit_ = num_fea_qubit_
        self.num_idx_qubit_ = num_idx_qubit_
        self.target_dim_ = 2
        self.CNOT = CNOT_layer(self.num_idx_qubit_ + self.num_fea_qubit_)

    def target_unitary(self, para_):
        U_layer = np.eye(2 ** (self.num_fea_qubit_ + self.num_idx_qubit_))
        total_q = self.num_idx_qubit_ + self.num_fea_qubit_
        for dep_ in range(self.depth_):
            U_temp_2 = np.eye(2)
            for i_th in range(total_q):
                if i_th == 0:
                    U_temp_2 = RY(para_[0 +  total_q * dep_])
                else:
                    U_temp_2 = np.kron(RY(para_[i_th +  total_q * dep_]), U_temp_2)

            U_layer = np.kron(CNOT(), I()) @ np.kron(I(), CNOT())  @ U_temp_2  @ U_layer

        return U_layer

    def compute_predict_vector(self, input_state_, para_, dep_, num_mea):
        r''' Given input state, output the predict vector with dimension self.target_dim_
        :param input_state_: the initial quantum state
        :param para_: the parameters to be updated
        :param dep_: depolarization rate
        :param num_mea: number of measurement
        '''
        ceil_dim = 2 ** int(np.ceil(np.log2(self.target_dim_)))
        predict_vec = np.zeros(shape=(self.target_dim_))

        # The dimension of the identity opeartor used in POVM
        dim_identify = int(len(input_state_) / ceil_dim)
        I_measure = I()
        # compute the ouput state
        unitary = self.target_unitary(para_)
        output_state = unitary @ input_state_
        rho =  np.outer(output_state.conj(), output_state)
        # depolarization channel
        rho = (1- dep_)*rho + dep_ * np.eye(len(output_state)) / len(output_state)
        # Start to measure
        basis_0 = np.zeros(shape=(2,2))
        basis_0[0,0] =1
        povm = np.kron(np.eye(4), basis_0)
        out = np.trace(povm @ rho)
        prob_dis = [out, 1- out]
        sample_res = np.random.choice(2, num_mea, p=prob_dis)
        res_0 = num_mea - np.count_nonzero(sample_res)
        prob_est = res_0 / num_mea
        return prob_est

def loss_func(mea, label):
    label = int(label)
    loss =  (label - mea) ** 2
    return loss

def loss_func_grad(mea, label):
    out = mea - label
    return out
