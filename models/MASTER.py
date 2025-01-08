import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.KAN import KAN
from layers.Ts2Vec import TSEncoder


class MultiScaleDecomp:
    def __init__(self, moving_avg_scales):
        """
        多尺度分解类
        :param moving_avg_scales: list[int] 不同时间窗口的滑动平均
        """
        self.moving_avg_scales = moving_avg_scales

    def decompose(self, x):
        """
        多尺度分解
        :param x: 输入时间序列, 形状 [batch_size, seq_len, feature_dim]
        :return: (long_trend, short_trend), 分别为长趋势和短趋势
        """
        long_trend = []
        short_trend = x  # 初始短趋势等于原始序列

        for scale in self.moving_avg_scales:
            avg = self.moving_average(x, scale)
            long_trend.append(avg)
            short_trend = short_trend - avg  # 减去长趋势分量

        # 汇总所有长趋势
        long_trend = torch.stack(long_trend, dim=-1)  # [batch_size, seq_len, feature_dim, len(moving_avg_scales)]
        return long_trend, short_trend

    @staticmethod
    def moving_average(x, window_size):
        """
        滑动平均计算
        :param x: 输入时间序列, 形状 [batch_size, seq_len, feature_dim]
        :param window_size: 滑动窗口大小
        :return: 平滑后的时间序列, 同样形状
        """
        padding = window_size // 2
        x_padded = F.pad(x, (0, 0, padding, padding - (1 if window_size % 2 == 0 else 0)), mode='reflect')
        kernel = torch.ones(x.size(-1), 1, window_size, device=x.device) / window_size
        avg = F.conv1d(x_padded.permute(0, 2, 1), kernel, groups=x.size(-1)).permute(0, 2, 1)
        return avg


class StageEncoding(nn.Module):
    '''
    发酵阶段编码，根据不同的发酵时期划分各个阶段
    '''
    def __init__(self):
        super(StageEncoding, self).__init__()

    def forward(self, hh):
        # hh: Shape [batch_size, seq_len] 根据发酵阶段划分专家系统MoE
        stage = torch.zeros(hh.size(0), hh.size(1), 4, device=hh.device)
        stage[:, :, 0] = (hh < 40).float()
        stage[:, :, 1] = ((hh >= 40) & (hh < 60)).float()
        stage[:, :, 2] = ((hh >= 60) & (hh < 120)).float()
        stage[:, :, 3] = (hh >= 120).float()
        return stage  # Shape [batch_size, seq_len, 4]


class ContinuousStageEncoding(nn.Module):
    def __init__(self, max_time=200, num_frequencies=4):
        """
        Continuous encoding of fermentation stages using sinusoidal functions.
        :param max_time: Maximum fermentation time for normalization.
        :param num_frequencies: Number of frequency components to encode.
        """
        super(ContinuousStageEncoding, self).__init__()
        self.max_time = max_time
        self.num_frequencies = num_frequencies  # Number of sinusoidal components

    def forward(self, hh):
        """
        :param hh: Shape [batch_size, seq_len], fermentation time in hours.
        :return: Shape [batch_size, seq_len, num_frequencies * 2], sinusoidal encoding.
        """
        # Normalize fermentation time to [0, 1]
        normalized_time = hh / self.max_time  # Shape: [batch_size, seq_len]

        # Create sinusoidal features
        frequencies = torch.arange(1, self.num_frequencies + 1, device=hh.device).float()  # Shape: [num_frequencies]
        angles = normalized_time.unsqueeze(-1) * frequencies * torch.pi  # Shape: [batch_size, seq_len, num_frequencies]

        # Compute sin and cos components
        sin_features = torch.sin(angles)  # Shape: [batch_size, seq_len, num_frequencies]
        cos_features = torch.cos(angles)  # Shape: [batch_size, seq_len, num_frequencies]

        # Concatenate sin and cos components
        encoding = torch.cat([sin_features, cos_features], dim=-1)  # Shape: [batch_size, seq_len, num_frequencies * 2]
        return encoding


class MS_Dlinear_Block(nn.Module):
    def __init__(self, seq_len, pred_len, moving_avg_scales, max_time=200, num_frequencies=4):
        super(MS_Dlinear_Block, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 多尺度分解模块
        self.decomposition = MultiScaleDecomp(moving_avg_scales)

        # 每个长趋势分量的线性预测模块
        self.Linear_LongTrends = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in moving_avg_scales]
        )
        # 短趋势的线性预测模块
        self.Linear_ShortTrend = nn.Linear(seq_len, pred_len)
        # 发酵阶段编码模块
        # self.stage_encoding = ContinuousStageEncoding(max_time=max_time, num_frequencies=num_frequencies)

        # # 动态权重调整
        # self.dynamic_long_weights = nn.Sequential(
        #     nn.Linear(num_frequencies * 2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, len(moving_avg_scales)),  # 对应 long_trend 的每个尺度权重
        #     nn.Softmax(dim=-1)  # 归一化为概率分布
        # )
        # self.dynamic_short_weight = nn.Sequential(
        #     nn.Linear(num_frequencies * 2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),  # 单个权重
        #     nn.Sigmoid()  # 确保在 [0, 1] 区间
        # )

    def forward(self, x, x_mark_enc):
        long_trend, short_trend = self.decomposition.decompose(x)  # 分解
        long_trend = long_trend.permute(0, 2, 1, 3)
        short_trend = short_trend.permute(0, 2, 1)
        # 对每个长趋势分量应用独立的线性模型
        long_trend_outputs = []
        for i, linear in enumerate(self.Linear_LongTrends):
            trend_i = long_trend[..., i]  # 提取第 i 个长趋势分量
            long_trend_outputs.append(linear(trend_i))

        long_trend_output = sum(long_trend_outputs)  # 汇总所有长趋势预测
        # long_trend_output = torch.stack(long_trend_outputs, dim=-1)

        # 对短趋势分量进行预测
        short_trend_output = self.Linear_ShortTrend(short_trend)

        # hh = x_mark_enc[:, :, -1]
        # stage_features = self.stage_encoding(hh)
        # stage_features_mean = stage_features.mean(dim=1)
        # long_weights = self.dynamic_long_weights(stage_features_mean)  # [batch_size, num_scales]
        # long_weights = long_weights.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, num_scales]
        # short_weight = self.dynamic_short_weight(stage_features_mean)  # [batch_size, 1]
        # short_weight = short_weight.unsqueeze(-1)  # [batch_size, 1, 1]
        # 合并预测结果
        output = long_trend_output + short_trend_output
        # weighted_long_trend = (long_weights * long_trend_output).sum(dim=-1)
        # output = (1 - short_weight) * weighted_long_trend + short_weight * short_trend_output

        output = output.permute(0, 2, 1)
        return output


class MoELayer(nn.Module):
    def __init__(self, num_experts, seq_len, pred_len, feature_dim, moving_avg, top_k, kl_lambda=0.001):
        super(MoELayer, self).__init__()
        self.moving_avg_scales = [3, 7, 14]
        # self.experts = nn.ModuleList([Dlinear_Block(seq_len, pred_len, moving_avg) for _ in range(num_experts)])
        self.experts = nn.ModuleList([MS_Dlinear_Block(seq_len, pred_len, self.moving_avg_scales) for _ in range(num_experts)])
        self.gate = nn.Linear(feature_dim + 2 * 4, num_experts)  # Gate works per feature channel
        self.pred_len = pred_len
        self.top_k = top_k
        self.stage_encoder = ContinuousStageEncoding(max_time=200, num_frequencies=4)
        self.kl_lambda = kl_lambda

    def forward(self, x, x_mark_enc):
        # x shape: [batch_size, seq_len, feature_dim] x_mark_enc: [batch_size, seq_len, 4]
        hh = x_mark_enc[:, :, -1]
        # stage_encoding = StageEncoding()(hh)
        stage_encoding = self.stage_encoder(hh)
        x_cat = torch.cat([x, stage_encoding], dim=-1)

        gate_score = F.softmax(self.gate(x_cat), dim=-1)  # `gate_score`: [batch_size, seq_len, num_experts]
        gate_score = gate_score[:, -self.pred_len:, :]

        # Compute KL divergence to encourage uniform distribution
        uniform_distribution = torch.full_like(gate_score, 1 / gate_score.size(-1))
        kl_div = self.kl_lambda * F.kl_div(gate_score.log(), uniform_distribution, reduction='batchmean')

        top_k_values, top_k_indices = torch.topk(gate_score, self.top_k, dim=-1)  # Shape: [batch_size, pred_len, top_k]
        mask = torch.zeros_like(gate_score).scatter_(-1, top_k_indices, top_k_values)  # Mask non-top-k elements
        gate_score = mask / mask.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x, x_mark_enc) for expert in self.experts], dim=-1)  # Shape: [batch_size, pred_len, feature_dim, num_experts]

        gate_score = gate_score.unsqueeze(2)  # Shape: [batch_size, pred_len, 1, num_experts]
        output = (expert_outputs * gate_score).sum(dim=-1)  # Shape: [batch_size, pred_len, feature_dim]

        return output, kl_div


class Model(nn.Module):
    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len + configs.pred_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.num_experts = configs.num_experts
        self.feature_dim = configs.enc_in
        self.moving_avg = configs.moving_avg
        self.top_k = configs.top_k
        self.MoELayer = MoELayer(self.num_experts, self.seq_len, self.pred_len, self.feature_dim, self.moving_avg, self.top_k)
        self.pre_linear = nn.Linear(self.feature_dim, configs.c_out)
        # self.pre_gnn = GNNForecaster(self.feature_dim, self.feature_dim)

    def forecast(self, x_enc, x_mark_enc):
        x_enc, kl_div = self.MoELayer(x_enc, x_mark_enc)
        output = self.pre_linear(x_enc)
        return output, kl_div

    def imputation(self, x_enc):
        return x_enc

    def anomaly_detection(self, x_enc):
        return x_enc

    def classification(self, x_enc):
        return x_enc

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, kl_div = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :], kl_div  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
