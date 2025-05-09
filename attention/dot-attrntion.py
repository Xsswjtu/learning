import numpy as np
from scipy.special import softmax

def dot_product_attention(query, key, value, mask=None):
    """
    简单的点积注意力机制。

    Args:
        query: 查询向量，形状为 (batch_size, num_queries, query_dim)
        key: 键向量，形状为 (batch_size, num_keys, key_dim)
        value: 值向量，形状为 (batch_size, num_keys, value_dim)
        mask: 可选的注意力掩码，形状为 (batch_size, num_queries, num_keys)，
              用于屏蔽某些键值对。

    Returns:
        output: 注意力加权的值向量，形状为 (batch_size, num_queries, value_dim)
        attention_weights: 注意力权重，形状为 (batch_size, num_queries, num_keys)
    """
    # 1. 计算 Query 和 Key 的点积，得到注意力分数 (logits)
    #    形状: (batch_size, num_queries, key_dim) @ (batch_size, key_dim, num_keys)
    #          -> (batch_size, num_queries, num_keys)
    attention_logits = np.matmul(query, np.transpose(key, axes=(0, 2, 1)))

    # 2. 缩放注意力分数
    #    除以 key 的维度 (key_dim) 的平方根，以稳定训练
    dk = key.shape[-1]
    scaled_attention_logits = attention_logits / np.sqrt(dk)

    # 3. 应用掩码 (如果提供)
    if mask is not None:
        # 将掩码为 1 的位置设置为负无穷，这样在 softmax 后权重会趋近于 0
        scaled_attention_logits = np.where(mask == 0, -1e9, scaled_attention_logits)

    # 4. 计算注意力权重 (softmax)
    #    对最后一个维度 (num_keys) 进行 softmax 归一化
    attention_weights = softmax(scaled_attention_logits, axis=-1)

    # 5. 将注意力权重应用于 Value 向量
    #    形状: (batch_size, num_queries, num_keys) @ (batch_size, num_keys, value_dim)
    #          -> (batch_size, num_queries, value_dim)
    output = np.matmul(attention_weights, value)

    return output, attention_weights

if __name__ == '__main__':
    # 示例数据
    batch_size = 2
    num_queries = 3
    num_keys = 4
    query_dim = 5
    key_dim = 5
    value_dim = 6

    # 随机生成 Query, Key, Value 向量
    query = np.random.rand(batch_size, num_queries, query_dim)
    key = np.random.rand(batch_size, num_keys, key_dim)
    value = np.random.rand(batch_size, num_keys, value_dim)

    # 创建一个简单的掩码 (可选)
    mask = np.array([
        [[1, 1, 0, 1],
         [1, 0, 1, 1],
         [1, 1, 1, 0]],
        [[1, 0, 1, 1],
         [1, 1, 0, 1],
         [0, 1, 1, 1]]
    ])

    # 计算点积注意力
    output, attention_weights = dot_product_attention(query, key, value, mask)

    print("Query 形状:", query.shape)
    print("Key 形状:", key.shape)
    print("Value 形状:", value.shape)
    print("掩码 形状:", mask.shape if mask is not None else None)
    print("\n输出 (Attention Output) 形状:", output.shape)
    print("\n注意力权重 (Attention Weights) 形状:", attention_weights.shape)

    print("\n第一个批次的注意力权重:")
    print(attention_weights[0])