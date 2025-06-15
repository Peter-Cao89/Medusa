import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.
    1. 存储历史key/value
    2. 动态扩展缓存长度
    3. 支持高效的数据复制与拼接
    4. 维护当前有效缓存长度

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.存储keys/values的张量
        current_length (int): Current length of the data being stored.当前存储数据的长度
    """

    def __init__(
            self,
            data: torch.Tensor,
            current_length: int):
        """
        Initialize the KVCache.初始化kvcache。

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],  # batch_size
            self.data.shape[1],  # num_heads
            self.current_length.item(),  # 实际使用长度(非预分配长度)
            self.data.shape[3],  # head_dim
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.
        将指定索引的数据复制到缓存的新位置(用于 缓存滚动更新)。
        使用场景：比如在MEDUSA解码中，需要根据不同的候选路径选择性的"继承"某些tokens的缓存信息。

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.需要复制的索引（如候选路径对应的 token 索引）
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2. 默认是第2维（即序列长度维度）
        """
        tgt = self.data.index_select(dim, indices)  # 从当前 cache 中选出特定位置的数据
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])  # 在目标位置预留空间
        dst.copy_(tgt, non_blocking=True)  # 异步复制数据
        self.current_length.fill_(prev_length + tgt.shape[dim])  # 更新长度

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    # 将batch size初始化为1,可以针对不同的batch size进行修改。
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers
    # 初始化一个tensor，用于为所有的层存储历史keys与values。shape为[2×num_layers, batch_size, kv_heads, max_len, head_dim]
    past_key_values_data = torch.zeros(
        config.num_hidden_layers * 2,  # 层数，每层有 key + value => ×2
        batch_size,
        config.num_key_value_heads,  # KV heads 数量（可能≠注意力头数）
        config.max_position_embeddings,  # 最大序列长度
        config.hidden_size // config.num_attention_heads,  # 每个头的维度
        device=model.device,
        dtype=model.dtype,
    )
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j],
                        current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data
