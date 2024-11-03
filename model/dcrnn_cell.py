from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from lib import utils


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation  # 激活函数，默认为 tanh，用于对更新后的状态进行非线性变换。
        self._num_nodes = num_nodes  # 图中节点的总数。
        self._num_proj = num_proj  # 输出的投影维度，若为 None 则不会进行投影操作。
        self._num_units = num_units  # 指定 GRU 隐藏层的单元数（隐藏状态的维度），即每个节点在 GRU 中的输出维度。
        self._max_diffusion_step = max_diffusion_step  # 最大扩散步数，决定了图卷积的邻域范围，即在图卷积操作中使用的步数。
        self._supports = []  # 生成图卷积矩阵
        self._use_gc_for_ru = use_gc_for_ru  # 布尔值，表示是否在 GRU 的重置门和更新门中使用图卷积操作。
        supports = []
        if filter_type == "laplacian":  # 计算归一化拉普拉斯矩阵
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":  # 计算随机游走矩阵的转置
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":  # 计算双重随机游走矩阵，即邻接矩阵和其转置的随机游走矩阵
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:  # 默认使用拉普拉斯矩阵
            supports.append(utils.calculate_scaled_laplacian(adj_mx))

        # 转换为 TensorFlow 的 SparseTensor 格式，以便高效地进行稀疏矩阵运算
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    # 前向传播逻辑
    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):  # 添加变量的作用域（前缀）
            # 1.计算更新门u和重置门r
            with tf.variable_scope("gates"):
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                # 判断使用哪种方法计算更新门和重置门
                if self._use_gc_for_ru:
                    fn = self._gconv
                else:
                    fn = self._fc
                value = tf.nn.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
                # 拆分并调整重置门和更新门的形状
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))

            # 2.计算候选状态 c
            with tf.variable_scope("candidate"):
                c = self._gconv(inputs, r * state, self._num_units)
                if self._activation is not None:
                    c = self._activation(c)

            # 3. 计算输出和新状态
            output = new_state = u * state + (1 - u) * c

            # 4.可选的输出投影层: 调整输出的维度, 节省内存并简化后续操作
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    # 普通全连接
    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    # 扩散卷积
    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        # 1.合并输入和状态
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)  # 将当前输入和上一个时间步的状态一起传入图卷积。
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        # 2.初始化图卷积
        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:  # 根据 _max_diffusion_step 控制扩散层数，0 表示无扩散
                pass
            else:
                for support in self._supports:
                    # 3.将 support（稀疏邻接矩阵）与 x0（或更新的 x1）相乘，模拟信息在图上扩散。
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    # 4.将扩散结果 x1、x2 依次拼接到 x 上
                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0  # 切比雪夫多项式算法
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            # 5.合并扩散结果：(batch_size * num_nodes, input_size * num_matrices)
            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            # 6.应用权重和偏置，得到卷积输出
            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
