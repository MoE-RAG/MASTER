from data_provider.data_factory import data_provider
from models import Autoformer, DLinear, TimeLLM, TimesNet, PatchTST, Informer, iTransformer, \
    TimeXer, TiDE, MtsLLM, woLLM, TLinear, TimeModeLLM
from tqdm import tqdm
import time as t
import torch
from utils.losses import normalized_temporal_smoothness_loss
from torch import nn, optim
import argparse
import os
import random
import numpy as np
import pandas as pd
from utils.tools import validate
from utils.metrics import metric
import matplotlib.pyplot as plt
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Time-LLM')
# basic config
parser.add_argument('--model', type=str, default='TimeModeLLM',
                    help='model name, options: [TimeLLM, Autoformer, DLinear, TLinear, TimesNet, PatchTST, Informer, iTransformer'
                         'TimeXer, TiDE, MtsLLM, woLLM, TimeModeLLM]')
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# data loader
parser.add_argument('--data', type=str, default='EFP_RAG', help='dataset type')
parser.add_argument('--root_path', type=str, default='F:\Time-LLM\Time-LLM-main\dataset\EFP', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='EFP_long.csv', help='data file')
parser.add_argument('--target', type=str, default='hx', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S: univariate predict univariate, MS:multivariate predict univariate')
# model define
parser.add_argument('--enc_in', type=int, default=24, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

parser.add_argument('--num_experts', type=int, default=5, help='num_experts')
parser.add_argument('--top_k', type=int, default=2, help='num_experts')
parser.add_argument('--moving_avg_scales', type=int, nargs='+', default=[3, 7, 14], help='List of moving average scales (e.g., --moving_avg_scales 3 5 7)')

parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
# parser.add_argument('--top_k', type=int, default=2, help='TimesNet top_k')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method, only support avg, max, conv')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=4, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--llm_layers', type=int, default=6)

# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=24, help='start token length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

# optimization
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=32, help='train epochs')
parser.add_argument('--num_experiments', type=int, default=2, help='Number of training sessions')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()


def main_run(model, optimizer, train_data, train_loader, vali_data, vali_loader, test_data, test_loader, num_experiments):
    time_now = t.time()
    train_steps = len(train_loader)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    prediciton_loss = []
    KL_loss = []
    TSmooth_loss = []
    t0 = t.time()
    for epoch in range(args.train_epochs):
        iter_count = 0
        pre_loss_count = 0
        kl_loss_count = 0
        tsmooth_loss_count = 0
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            with torch.cuda.amp.autocast():
                outputs, kl_div = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                pred_loss = criterion(outputs, batch_y)
                temp_smooth_loss = normalized_temporal_smoothness_loss(outputs, batch_y)
                total_loss = pred_loss + kl_div + 0.001 * temp_smooth_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                pre_loss_count += pred_loss.item()
                kl_loss_count += kl_div.item()
                tsmooth_loss_count += temp_smooth_loss.item()



            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | pred_loss: {2:.7f} KL_loss: {3:.7f} TSom_loss: {4:.7f}".format(i + 1, epoch + 1, pred_loss.item(), kl_div.item(), temp_smooth_loss.item()))
                speed = (t.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = t.time()

        prediciton_loss.append(pre_loss_count / i)
        KL_loss.append(kl_loss_count / i)
        TSmooth_loss.append(tsmooth_loss_count / i)
        vali_loss, vali_mae_loss = validate(args, model, vali_data, vali_loader, criterion, mae_metric)
        print("Epoch: {0} | Vali Loss: {1:.7f} MAE Loss: {2:.7f}".format(epoch + 1, vali_loss, vali_mae_loss))

    t1 = t.time()
    train_time = t1 - t0

    loss_data = {
        "pred_loss": prediciton_loss,
        "KL_loss": KL_loss,
        "TSmooth_loss": TSmooth_loss
    }
    loss_df = pd.DataFrame(loss_data)
    # 将 DataFrame 保存为 Excel 文件
    loss_df.to_csv(f"./results\RAG/loss {args.model}_{args.seq_len}_{args.pred_len}_{num_experiments}.csv", index=False)
# 训练集情况绘制
    train_predictions = []
    train_truths = []
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不需要梯度更新
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # 推理模型
            outputs, kl_div = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, :]  # 预测值
            batch_y = batch_y[:, -args.pred_len:, :]  # 真实值

            # 保存预测值和真实值
            train_predictions.append(outputs.detach().cpu().numpy())
            train_truths.append(batch_y.detach().cpu().numpy())

    # 转换为 NumPy 数组
    train_predictions = np.concatenate(train_predictions, axis=0)
    train_truths = np.concatenate(train_truths, axis=0)

    # 如果是单变量预测（假设特征维度=1）
    train_pre_tocsv = train_predictions[:, :, 0]  # 去掉特征维度 (样本数, 预测步长)
    train_truths_tocsv= train_truths[:, :, 0]

    # 转换为 DataFrame
    pred_df = pd.DataFrame(train_pre_tocsv, columns=[f'Pred_Step_{i + 1}' for i in range(train_pre_tocsv.shape[1])])
    truth_df = pd.DataFrame(train_truths_tocsv, columns=[f'Truth_Step_{i + 1}' for i in range(train_truths_tocsv.shape[1])])

    # 添加样本索引
    pred_df.insert(0, 'Sample_Index', np.arange(len(train_pre_tocsv)))
    truth_df.insert(0, 'Sample_Index', np.arange(len(train_truths_tocsv)))

    # 保存到 CSV 文件
    pred_df.to_csv(f'./results\RAG\Train_pre_{args.model}_EFP_RAG_{args.seq_len}_{args.pred_len}_{num_experiments}.csv', index=False)
    truth_df.to_csv(f'./results\RAG\Train_truth_{args.model}_EFP_RAG_{args.seq_len}_{args.pred_len}_{num_experiments}.csv', index=False)

    print("训练集的预测值和真实值已保存")

    num_samples_to_plot = 10
    pred_to_plot = train_predictions[:num_samples_to_plot, :, 0].flatten()
    truth_to_plot = train_truths[:num_samples_to_plot, :, 0].flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(pred_to_plot, label='Predictions', color='red', linestyle='--')
    plt.plot(truth_to_plot, label='Ground Truth', color='blue', linestyle='-')
    plt.legend()
    plt.title("Final Training Predictions vs Ground Truth")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.savefig(f'./results\RAG\Train_{args.model}_EFP_RAG_{args.seq_len}_{args.pred_len}_{num_experiments}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


    predict = []
    truth = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            with torch.cuda.amp.autocast():
                outputs, kl_div = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]

            pred = outputs.detach()
            true = batch_y.detach()
            predict.append(pred.cpu().numpy())
            truth.append(true.cpu().numpy())
    # 测试集正确率
    pre_total = np.concatenate(predict, axis=0)  # 将所有批次的预测值拼接起来
    true_total = np.concatenate(truth, axis=0)  # 将所有批次的真实值拼接起来

    # 如果是单变量预测（假设特征维度=1）
    test_pre_tocsv = pre_total[:, :, 0]  # 去掉特征维度 (样本数, 预测步长)
    test_truths_tocsv= true_total[:, :, 0]

    # 转换为 DataFrame
    pred_df = pd.DataFrame(test_pre_tocsv, columns=[f'Pred_Step_{i + 1}' for i in range(test_pre_tocsv.shape[1])])
    truth_df = pd.DataFrame(test_truths_tocsv, columns=[f'Truth_Step_{i + 1}' for i in range(test_truths_tocsv.shape[1])])

    # 添加样本索引
    pred_df.insert(0, 'Sample_Index', np.arange(len(test_pre_tocsv)))
    truth_df.insert(0, 'Sample_Index', np.arange(len(test_truths_tocsv)))

    # 保存到 CSV 文件
    pred_df.to_csv(f'./results\RAG\Test_pre_{args.model}_EFP_RAG_{args.seq_len}_{args.pred_len}_{num_experiments}.csv', index=False)
    truth_df.to_csv(f'./results\RAG\Test_truth_{args.model}_EFP_RAG_{args.seq_len}_{args.pred_len}_{num_experiments}.csv', index=False)

    print("测试集的预测值和真实值已保存")

    predict_p = np.array(pre_total)  # 转换为 numpy 数组，形状应为 [3800, 128, 1]
    truth_p = np.array(true_total)  # 转换为 numpy 数组，形状应为 [3800, 128, 1]

    # 取前10个样本
    num_samples = 10
    predict_samples = predict_p[:num_samples].squeeze(-1)  # [10, 128]
    truth_samples = truth_p[:num_samples].squeeze(-1)  # [10, 128]

    # 前后拼接
    time_steps = predict_samples.shape[1]
    x_axis = np.arange(time_steps * num_samples)  # 总时间步
    predict_concat = predict_samples.flatten()  # 拼接预测值
    truth_concat = truth_samples.flatten()  # 拼接真实值

    # 绘图
    plt.figure(figsize=(15, 6))
    plt.plot(x_axis, truth_concat, label='True Values', color='blue', linewidth=1)
    plt.plot(x_axis, predict_concat, label='Predicted Values', color='red', linestyle='--', linewidth=1)

    # 标注样本分割线
    for i in range(1, num_samples):
        plt.axvline(x=i * time_steps, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    plt.xlabel('Time Step (Concatenated Across Samples)')
    plt.ylabel('Value')
    plt.title(f'{args.model}_{args.seq_len}_{args.pred_len}_{num_samples}')
    plt.legend()

    # 保存图像
    plt.savefig(f'./results\RAG\Test_{args.model}_EFP_RAG_{args.seq_len}_{args.pred_len}_{num_experiments}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    pre_total = pre_total.reshape(-1, 1)
    true_total = true_total.reshape(-1, 1)

    MAE, MSE, RMSE, MAPE, MSPE, SMAPE, MASE = metric(pre_total, true_total)

    print("MAE : %f" % MAE)
    print("MSE : %f" % MSE)
    print("RMSE : %f" % RMSE)
    print("MAPE : %f" % MAPE)
    print("MSPE: %f" % MSPE)
    print("SMAPE: %f" % SMAPE)
    print("MASE: %f" % MASE)
    print("训练时间：%f" % train_time)

    print(MAE)
    print(MSE)
    print(RMSE)
    print(MAPE)
    print(MSPE)
    print(SMAPE)
    print(MASE)
    print(train_time)
    return MAE, MSE, RMSE, MAPE, MSPE, SMAPE, MASE, train_time


if __name__ == "__main__":
    seed = random.randint(0, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    results = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "MAPE": [],
        "MSPE": [],
        "SMAPE": [],
        "MASE": [],
        "Time": []
    }
    for i in range(args.num_experiments):
        print(f"Running experiment {i + 1}")
        # 假设 main_run() 返回一个包含多个指标的元组
        if args.model == 'TimeLLM':
            model = TimeLLM.Model(args).float().to(device)
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float().to(device)
        elif args.model == 'Autoformer':
            model = Autoformer.Model(args).float().to(device)
        elif args.model == 'TimesNet':
            model = TimesNet.Model(args).float().to(device)
        elif args.model == 'PatchTST':
            model = PatchTST.Model(args).float().to(device)
        elif args.model == 'Informer':
            model = Informer.Model(args).float().to(device)
        elif args.model == 'iTransformer':
            model = iTransformer.Model(args).float().to(device)
        elif args.model == 'TimeXer':
            model = TimeXer.Model(args).float().to(device)
        elif args.model == 'TiDE':
            model = TiDE.Model(args).float().to(device)
        elif args.model == 'MtsLLM':
            model = MtsLLM.Model(args).float().to(device)
        elif args.model == 'woLLM':
            model = woLLM.Model(args).float().to(device)
        elif args.model == 'TLinear':
            model = TLinear.Model(args).float().to(device)
        elif args.model == 'TimeModeLLM':
            model = TimeModeLLM.Model(args).float().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        mae, mse, rmse, mape, mspe, smape, mase, time = main_run(model, optimizer, train_data, train_loader, vali_data,
                                                                 vali_loader, test_data, test_loader, i)

        # 使用循环来添加结果
        for key, value in zip(results.keys(), [mae, mse, rmse, mape, mspe, smape, mase, time]):
            results[key].append(value)
        print(f'Saved {i + 1}')

    print("All experiment results:", results["MAE"])

    # 将结果保存为 CSV 文件
    df = pd.DataFrame(results).T
    df.to_csv(f"./results/RAG/{args.model}_EFPeach_{args.seq_len}-{args.label_len}-{args.pred_len}.csv", index=False)
    print("Results saved to csv")