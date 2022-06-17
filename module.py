import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from itertools import product

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


def construct_model(input_para):
    model = keras.Sequential()
    model.add(Dense(128, input_dim=input_para, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss='mse')

    return model


def train_model(model, X, Y, validation=True, label = 0, dir = './'):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Training PCT prediction model, trial = ' + str(label))
    print(X[0])
    print(Y[0])
    X = (X + 5) / 10
    Y = Y/100
    batch_size = 256

    if validation:
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=1000)
        cb_checkpoint = ModelCheckpoint(filepath = dir + 'model_' + str(label) + '.h5', monitor='val_loss', verbose=0,
                                        save_best_only=True)
        model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 1000000, callbacks=[cb_checkpoint, early_stopping], batch_size=batch_size, verbose=0)
    else:
        early_stopping = EarlyStopping(monitor='loss', patience = 100)
        cb_checkpoint = ModelCheckpoint(filepath = dir + 'model_' + str(label) + '.h5', monitor='loss', verbose=1,
                                        save_best_only=True)
        model.fit(X, Y, epochs = 1000000, callbacks=[cb_checkpoint, early_stopping], batch_size=batch_size, verbose=0)

    return model

# 개선 필요
def get_uncertainty(model, point, mc = 2000):
    x = []
    for i in range(mc):
        x.append((point + 5)/10)
    x = np.array(x)
    PCTs = model(x, training=True).numpy() * 100

    return np.mean(PCTs), np.var(PCTs), np.std(PCTs)


def all_branch_uncertainty(model, all_cases, mc=2000):
    # Find uncertainty for every branches
    print('-----------------------------------------------------------------------------------------------------------')
    print('Find uncertainty with Monte-carlo dropout, MC = ' + str(mc))
    
    sum = model((all_cases + 5)/10, training=True).numpy() * 100
    square_sum = np.square(sum)

    for i in range(mc-1):
        print('\r' + str(np.round_(i/mc*100, 2)) + ' %', end='')
        temp = model((all_cases + 5)/10, training=True).numpy() * 100
        sum = sum + temp
        square_sum = square_sum + np.square(temp)


    mean = sum / mc
    var = square_sum / mc - np.square(mean)
    std = np.sqrt(var)
    '''
    result = []
    length = int(len(all_cases))
    for idx, case in enumerate(all_cases):
        print('\r' + str(np.round_(idx/length*100, 2)) + ' %', end='')
        mean, var, std = get_uncertainty(model, case, mc = mc)
        result.append([mean, var, std])
    '''
    result = np.hstack((mean, var, std))


    return result


def standard_score_of_1478(uncertainty, n = 10, score = 1.95, label=0 ,dir ='./'):
    result =[]
    for info in uncertainty:
        result.append((90 - info[0])/(info[2]+1e-8))

    arranged = np.argsort(result)

    if result[arranged[-1]] < score:
        idxs = arranged[- n:]
    else:
        for end, idx in enumerate(arranged):
            if result[idx] > score:
                break
        for start, idx in enumerate(arranged):
            if result[idx] > - score:
                break
        idxs = np.random.choice(arranged[start:end], size=n)

    save = np.hstack((uncertainty, np.reshape(result, (int(len(result)), 1))))
    header = 'mean,var,std,1478_score'
    np.savetxt(dir + 'uncertainty_' + str(label) + '.csv', save, delimiter=',', header=header, fmt='%1.2f')

    return idxs, result


def standard_score_of_1478_all(uncertainty, score = 1.95, label=0, dir = ',/'):
    result =[]
    for info in uncertainty:
        result.append((90 - info[0])/(info[2]+1e-8))

    arranged = np.argsort(result)

    for end, idx in enumerate(arranged):
        if result[idx] > score:
            break
    for start, idx in enumerate(arranged):
        if result[idx] > - score:
            break

    idxs = arranged[start:end]

    save = np.hstack((uncertainty, np.reshape(result, (int(len(result)), 1))))
    header = 'mean,var,std,1478_score'
    np.savetxt(dir + 'uncertainty_' + str(label) + '.csv', save, delimiter=',', header=header, fmt='%1.2f')

    return idxs, result


# This code should be replaced into TH simulation control program
class Simulation():

    def __init__(self):
        data = open('data.csv', 'r', encoding='UTF-8')
        time = data.readline().split()
        input = []
        output = []
        for line in data.readlines():
            line = line.split()[0].replace(',,', '').split(',')
            input.append(np.array([float(i) for i in line[:6]]))
            maximum = max([float(i) for i in line[6:]])
            output.append(np.array([maximum]))

        self.x = np.array(input)
        self.y = np.array(output)


    def simulation(self, branch):
        branch = np.array(branch)

        '''
        for idx, x in enumerate(self.x):
            if sum(abs(x-branch)) < 0.001:

                result = self.y[idx]
        '''

        idx = np.squeeze(np.where((branch[0] == self.x[:, 0])
                                        & (branch[1] == self.x[:, 1])
                                        & (branch[2] == self.x[:, 2])
                                        & (branch[3] == self.x[:, 3])
                                        & (branch[4] == self.x[:, 4])
                                        & (branch[5] == self.x[:, 5])))
        return self.y[idx]

simulator = Simulation()


def simulation(args, all_branches, visiting, X, Y):
    for idx in args:
        if visiting[idx] < 0.001:
            X = np.vstack((X, np.array([all_branches[idx]])))
            Y = np.vstack((Y, np.array([simulator.simulation(all_branches[idx])])))

    return np.array(X), np.array(Y)


def extreme_case_simulation(branch):

    items = []
    for axis in branch:
        items.append([axis[2][0], axis[2][-1]])

    basic_cases = list(product(*items))

    X = []
    Y = []
    for case in basic_cases:
        x = np.empty((0,))
        for i in case:
            x = np.hstack((x,i))
        y = simulator.simulation(x)
        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


def load_axis(file_name):
    temp = open(file_name, 'r')

    # 개행문자 삭제
    lines = []
    for line in temp.readlines():
        lines.append(line[:-1])
    branch = []
    for idx, line in enumerate(lines):
        if line == '*':
            name = str(lines[idx+1])
            str_case = []
            for idx_2, line_2 in enumerate(lines[idx+2:]):
                if line_2 == '*' or line_2 == '':
                    length = idx_2
                    break
                else:
                    str_case.append(line_2)

            case = []
            for node in str_case:
                node = node.split(',')
                case.append([float(i) for i in node])

            branch.append([name, length, case])
        elif line == '=':
            break

    input_para = 0
    for i in branch:
        input_para = input_para + int(len(i[2][0]))

    return branch, input_para


def load_branches(axis):
    # Based on axis info., find all possible branches
    items = []
    for ax in axis:
        items.append(ax[2])
    all_cases = list(product(*items))

    all_branches = []
    for case in all_cases:
        temp = []
        for i in case:
            temp = temp + i
        all_branches.append(temp)

    return np.array(all_branches)


def basic_case(axis, all_branches, random=0.05):

    number_of_branches = int(len(all_branches))

    X, Y = extreme_case_simulation(axis)
    print('Extreme case = ', int(len(X)))
    if random > 0.01:
        idx = np.random.choice(number_of_branches, int(number_of_branches * random), replace=False)
        print('Random case = ', int(len(idx)))
        X, Y = simulation(idx, all_branches, X, Y)

    return X, Y

class Critic():

    def __init__(self):
        self.real_PCT = simulator.y


    def record(self, all_branches, model, X, label=0 ,dir = './'):
        print('-----------------------------------------------------------------------------------------------------------')
        print('Record Start...., Iteration = ' + str(label))
        save = np.array(all_branches)
        save = np.hstack((save, self.real_PCT))
        print('Real PCT.....complete')

        pred_PCT = model.predict((all_branches + 5)/10) * 100

        # 아랫줄 수정 필요
        save = np.hstack((save, pred_PCT))
        
        print('Pred PCT.......complete')

        # 아래 코드 가속화 필요
        visiting = np.zeros(shape= (int(len(save)), 1))
        for idx, x in enumerate(all_branches):
            isin = np.squeeze(np.where((x[0] == X[:, 0])
                                            & (x[1] == X[:, 1])
                                            & (x[2] == X[:, 2])
                                            & (x[3] == X[:, 3])
                                            & (x[4] == X[:, 4])
                                            & (x[5] == X[:, 5])))
            if not(isin.size == 0):
                visiting[idx] = 1


        # 아랫줄 수정 필요
        save = np.hstack((save, visiting))
        print('Visiting check.......complete')


        header = 'Axis1,Axis2,Axis3,Axis4,Axis5,Axis6,real_PCT,pred_PCT,visiting'
        np.savetxt(dir + 'check_point_' + str(label) + '.csv', save, delimiter=',', header=header, fmt='%1.2f')

        return visiting, self.real_PCT, visiting