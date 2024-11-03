from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mae_loss
from model.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler  # 标准化

        # Train and loss
        self._loss = None
        self._mae = None  # 平均绝对误差
        self._train_op = None  # 训练操作

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))  # 控制扩散的步数，用于图卷积的范围。
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))  # 课程学习的衰减步数，用于调节模型学习真实标签的频率
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))  # RNN 层数
        rnn_units = int(model_kwargs.get('rnn_units'))  # 每层 RNN 中的隐藏单元数量
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))  # 布尔值，指示是否在训练中使用课程学习
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        # 用于接收输入数据和标签，通常在训练或测试时由实际数据填充
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        # GO_SYMBOL用于解码过程的起始输入，初始值为全零的张量
        # GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * input_dim))
        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))

        cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)

        # 创建多层RNN单元
        encoding_cells = [cell] * num_rnn_layers  # 在编码阶段将使用多个相同的RNN单元
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]  # 在解码的最后一层使用具有输出投影的单元，以确保输出维度正确。
        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        # 创建全局训练步数
        global_step = tf.train.get_or_create_global_step()

        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            # tf.unstack 用于分割一个列表，其中每个元素表示一个时间步的数据。
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            labels.insert(0, GO_SYMBOL)  # 将 GO_SYMBOL 作为解码器的初始输入插入标签列表的第一个位置

            # 控制每一步解码输入是使用模型的预测结果 prev，还是使用真实的标签值 labels[i]
            def _loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:  # 使用课程学习(模仿人类学习的特点，由简单到困难来学习课程)
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        # 基于全局步数 global_step 计算采样阈值 threshold
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        # 当随机数 c 小于 threshold 时，选择 labels[i]（真实值）；否则使用 prev（预测值）
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                return result

            # 构建编码器和解码器
            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels, enc_state, decoding_cells,
                                                              loop_function=_loop_function)

        # Project the output to output_dim.
        outputs = tf.stack(outputs[:-1], axis=1)  # 将解码器输出在时间维度上堆叠
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()  # 将所有的 TensorFlow summary 操作合并，便于在训练时记录模型的指标、损失和其他信息

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs
