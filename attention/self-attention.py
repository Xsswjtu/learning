import numpy as np
from scipy.special import softmax

def self_attention(input_sequence):
    """
    使用点积注意力实现简单的 Self-Attention。

    Args:
        input_sequence: 输入序列的 NumPy 数组，形状为 (batch_size, seq_len, feature_dim)。

    Returns:
        output: Self-Attention 的输出，形状为 (batch_size, seq_len, feature_dim)。
        attention_weights: 注意力权重，形状为 (batch_size, seq_len, seq_len)。
    """
    batch_size, seq_len, feature_dim = input_sequence.shape

    # 1. 定义线性变换的权重矩阵 (为了简化，这里直接使用随机矩阵)
    #    在实际应用中，这些权重矩阵是模型需要学习的参数
    W_Q = np.random.randn(feature_dim, feature_dim)
    W_K = np.random.randn(feature_dim, feature_dim)
    W_V = np.random.randn(feature_dim, feature_dim)

    # 2. 计算 Query, Key, Value
    #    形状: (batch_size, seq_len, feature_dim)
    Q = np.matmul(input_sequence, W_Q)
    K = np.matmul(input_sequence, W_K)
    V = np.matmul(input_sequence, W_V)

    # 3. 计算注意力分数 (点积)
    #    形状: (batch_size, seq_len, feature_dim) @ (batch_size, feature_dim, seq_len)
    #          -> (batch_size, seq_len, seq_len)
    attention_logits = np.matmul(Q, np.transpose(K, axes=(0, 2, 1)))

    # 4. 缩放注意力分数 (可选，但推荐)
    dk = feature_dim  # Key 的维度
    scaled_attention_logits = attention_logits / np.sqrt(dk)

    # 5. 计算注意力权重 (Softmax)
    #    对最后一个维度 (seq_len) 进行 softmax 归一化
    attention_weights = softmax(scaled_attention_logits, axis=-1)

    # 6. 将注意力权重应用于 Value
    #    形状: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, feature_dim)
    #          -> (batch_size, seq_len, feature_dim)
    output = np.matmul(attention_weights, V)

    return output, attention_weights

if __name__ == '__main__':
    # 示例输入序列
    batch_size = 2
    seq_len = 5
    feature_dim = 8
    input_seq = np.random.rand(batch_size, seq_len, feature_dim)

    # 计算 Self-Attention
    attention_output, weights = self_attention(input_seq)

    print("输入序列形状:", input_seq.shape)
    print("Self-Attention 输出形状:", attention_output.shape)
    print("注意力权重形状:", weights.shape)

    print("\n第一个批次的注意力权重:")
    print(weights[0])