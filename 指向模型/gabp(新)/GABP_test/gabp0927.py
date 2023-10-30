import chrom_code
import chrom_mutate
import chrom_cross
import chrom_select
import chrom_fitness
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import BP_network

fix_seed = 9
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class new_model(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output) -> None:
        super().__init__()
        self.dimensionality_dense = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_output),
            nn.Tanh())

        self.norm = nn.LayerNorm(n_feature)
        #nn.Dropout(0.01),
        self.dense = nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = self.dimensionality_dense(x)
        x = self.norm(x)
        x = self.dense(x)
        return x


def Train(model, dataset_x, dataset_y, test_x, test_y, EPOCH):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_x = torch.from_numpy(dataset_x).to(torch.float).to(device)
    train_y = torch.from_numpy(dataset_y.reshape(-1, 1)).to(torch.float).to(device)
    test_x = torch.from_numpy(test_x).to(torch.float).to(device)
    test_y = test_y
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # 组建数据开始训练
    model.to(device)
    torch_dataset = TensorDataset(train_x, train_y)
    BATCH_SIZE = 64
    model = model.train()
    train_loss = []
    print('Start training...')
    print('cuda(GPU)是否可用:', torch.cuda.is_available())
    print('torch的版本:', torch.__version__)
    for i in range(EPOCH):
        loader = DataLoader(dataset=torch_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
        temp_1 = []
        for step, (batch_x, batch_y) in enumerate(loader):
            out = model(batch_x)
            optimizer.zero_grad()
            loss = loss_fn(out, batch_y)
            temp_1.append(loss.item())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        train_loss.append(np.mean(np.array(temp_1)))
        if i % 10 == 0:
            print("The loss of {} epoch is {}".format(i, train_loss[-1]))
    return train_loss


if __name__ == "__main__":
    df = pd.read_excel('方位角数据.xlsx')
    data = np.array(df)
    np.random.shuffle(data)
    data_x = np.array(data)[:, 1:]
    data_y = np.array(data)[:, 0]
    N = data_x.shape[0]

    train_x = data_x[:int(N * 0.8), :]
    train_y = data_y[:int(N * 0.8)]
    test_x = data_x[int(N * 0.8):, :]
    test_y = data_y[int(N * 0.8):]
    print('数据准备完成...')

# 首先使用经典BP神经网络
    print("开始经典BP训练...")
    n_feature = 4
    n_hidden = 10
    n_output = 1
    EPOCH = 300
    learn_rate = 1e-2
    model = new_model(n_feature, n_hidden, n_output)
    BP_lossList = Train(model, train_x, train_y, test_x, test_y, EPOCH)
    print("经典BP训练完成...")


    print("开始进行遗传优化...")
    chrom_len = n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output  # 染色体长度
    size = 15  # 种群规模,用来轮盘赌选择
    bound = np.ones((chrom_len, 2))
    sz = np.array([[-1, 0], [0, 1]])
    bound = np.dot(bound, sz)  # 各基因取值范围
    p_cross = 0.4  # 交叉概率
    p_mutate = 0.01  # 变异概率
    maxgen = 30  # 遗传最大迭代次数

    chrom_sum = []  # 种群，染色体集合
    for i in range(size):
        chrom_sum.append(chrom_code.code(chrom_len, bound))
    account = 0  # 遗传迭代次数计数器
    best_fitness_ls = []  # 每代最优适应度
    ave_fitness_ls = []  # 每代平均适应度
    best_code = []  # 迭代完成适应度最高的编码值

    # 适应度计算
    fitness_ls = []
    for i in range(size):
        fitness = chrom_fitness.calculate_fitness(chrom_sum[i], n_feature, n_hidden, n_output,
                                                  EPOCH, learn_rate, train_x, train_y)
        fitness_ls.append(fitness)
    # 收集每次迭代的最优适应值和平均适应值
    fitness_array = np.array(fitness_ls).flatten()
    fitness_array_sort = fitness_array.copy()
    fitness_array_sort.sort()
    best_fitness = fitness_array_sort[-1]
    best_fitness_ls.append(best_fitness)
    ave_fitness_ls.append(fitness_array.sum() / size)

    while True:
        # 选择算子
        chrom_sum = chrom_select.select(chrom_sum, fitness_ls)
        # 交叉算子
        chrom_sum = chrom_cross.cross(chrom_sum, size, p_cross, chrom_len, bound)
        # 变异算子
        chrom_sum = chrom_mutate.mutate(chrom_sum, size, p_mutate, chrom_len, bound, maxgen, account + 1)
        # 适应度计算
        fitness_ls = []
        for i in range(size):
            fitness = chrom_fitness.calculate_fitness(chrom_sum[i], n_feature, n_hidden, n_output,
                                                      EPOCH, learn_rate, train_x, train_y)
            fitness_ls.append(fitness)
        # 收集每次迭代的最优适应值和平均适应值
        fitness_array = np.array(fitness_ls).flatten()
        fitness_array_sort = fitness_array.copy()
        fitness_array_sort.sort()
        best_fitness = fitness_array_sort[-1]  # 获取最优适应度值
        best_fitness_ls.append(best_fitness)
        ave_fitness_ls.append(fitness_array.sum() / size)
        # 计数器加一
        print(f"	第{account + 1}/{maxgen}次遗传迭代完成！")
        account = account + 1
        if account == maxgen:
            index = fitness_ls.index(max(fitness_ls))  # 返回最大值的索引
            best_code = chrom_sum[index]  # 通过索引获得对于染色体
            break

    # 参数提取
    hidden_weight = best_code[0:n_feature * n_hidden]
    hidden_bias = best_code[n_feature * n_hidden:
                            n_feature * n_hidden + n_hidden]
    output_weight = best_code[n_feature * n_hidden + n_hidden:
                              n_feature * n_hidden + n_hidden + n_hidden * n_output]
    output_bias = best_code[n_feature * n_hidden + n_hidden + n_hidden * n_output:
                            n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output]
    # 类型转换
    tensor_tran = transforms.ToTensor()
    hidden_weight = tensor_tran(np.array(hidden_weight).reshape((n_hidden, n_feature))).to(torch.float32)
    hidden_bias = tensor_tran(np.array(hidden_bias).reshape((1, n_hidden))).to(torch.float32)
    output_weight = tensor_tran(np.array(output_weight).reshape((n_output, n_hidden))).to(torch.float32)
    output_bias = tensor_tran(np.array(output_bias).reshape((1, n_output))).to(torch.float32)
    # 形状转换
    hidden_weight = hidden_weight.reshape((n_hidden, n_feature))
    hidden_bias = hidden_bias.reshape(n_hidden)
    output_weight = output_weight.reshape((n_output, n_hidden))
    output_bias = output_bias.reshape(n_output)
    GA = [hidden_weight, hidden_bias, output_weight, output_bias]

    gaBP_net = BP_network.GABP_net(n_feature, n_hidden, n_output, GA)
    gaBP_lossList = BP_network.train(gaBP_net, EPOCH, learn_rate, train_x, train_y)
    gaBP_prediction = gaBP_net(test_x).detach().numpy()
    print("遗传优化完成...")



    # 以下是数据展示部分
    torch.save(model, "model.pkl")
    model = model.eval()

    # BP模型误差下降曲线
    BP_prediction = model(test_x)
    BP_prediction = BP_prediction.cpu().data.numpy()
    plt.plot(BP_lossList)

    # 训练集结果对比
    plt.figure()
    train_res = model(train_x)
    train_res = train_res.cpu().data.numpy()
    plt.plot(train_res, label='train')
    plt.plot(train_y.cpu().data.numpy(), label='true')
    plt.legend()

    # 测试集验证结果对比
    plt.figure()
    plt.plot(test_y.flatten(), label='true')
    plt.plot(BP_prediction, label='pred')
    plt.legend()

    # 残差曲线
    plt.figure()
    Az_error = test_y.flatten() - BP_prediction.flatten()
    plt.plot(Az_error, label='Az_error')
    plt.legend()



    # 对两种算法的误差评价
    loss_fc = torch.nn.MSELoss(reduction="sum")
    y_test_ = tensor_tran(test_y).to(torch.float).reshape(test_y.shape[0], 1)
    BP_error = loss_fc(new_model(test_x), y_test_).detach().numpy()
    gaBP_error = loss_fc(gaBP_net(test_x), y_test_).detach().numpy()
    print("BP算法误差为：", BP_error, "\nGABP算法误差为：", gaBP_error)

    # 将算法结果写入log.txt #
    f = open('log.txt', 'a', encoding='UTF-8')
    f.write("神经网络拓扑结构为：" + str(n_feature) + ' ' + str(n_hidden) + ' ' + str(n_output) + '\n')
    f.write("网络迭代次数：" + str(EPOCH) + '\n')
    f.write("遗传迭代所获得的最优权值为：" + str(best_code) + "\n")
    f.write("改进算法预测值为\n" + str(gaBP_prediction.flatten()) + '\n')
    f.write(f"BP算法误差：{BP_error} \nGABP算法误差：{gaBP_error}\n\n")
    f.close()

    # 可视化 #
    plt.figure()
    plt.plot(BP_prediction, label='BP预测', c='r')
    plt.plot(test_y, label='真值', c='b')
    plt.grid(ls='--')
    plt.legend()

    plt.figure()
    plt.plot(BP_lossList, c='b')
    plt.ylabel("BP误差下降曲线")

    plt.figure()
    plt.plot(gaBP_lossList, c='b')
    plt.ylabel("GABP误差下降曲线")

    plt.figure()
    plt.plot(gaBP_prediction, label='GABP预测', c='r')
    plt.plot(test_y, label='真值', c='b')
    plt.grid(ls='--')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(BP_prediction.flatten(), test_y.flatten(), ls='', marker='o', color='#003C9D', markersize=5, alpha=0.2)
    min_, max_ = min(np.min(BP_prediction), np.min(test_y)), max(np.max(BP_prediction), np.max(test_y))
    plt.plot([min_, max_], [min_, max_], color='black', ls='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.show()

