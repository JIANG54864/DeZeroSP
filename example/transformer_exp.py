import numpy as np
from  dezerosp.transformer import Transformer

def example():
    """使用示例"""
    # 参数设置
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 10

    # 创建模型
    model = Transformer(src_vocab_size, tgt_vocab_size)

    # 创建示例数据
    src = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len)).astype(np.int32)
    tgt = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).astype(np.int32)

    # 前向传播
    output = model(src, tgt)

    print(f"输入源序列形状: {src.shape}")
    print(f"输入目标序列形状: {tgt.shape}")
    print(f"输出形状: {output.shape}")

    return model, output


if __name__ == "__main__":
    model, output = example()