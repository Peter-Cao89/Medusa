import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict
from medusa_model import MedusaModelABC, MedusaModelLlama, MedusaModelMistral

TOPK=10 # topk for sparse tree (10 is a placeholder and it is sufficient)

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(medusa_choices: list[list], device="cuda") -> Dict[str, torch.Tensor]:
    """
    Generate buffers for the Medusa structure based on the provided choices.基于输入的medusa choices为medusa结构生成各种buffers，
    包括medusa_attention_mask, medusa_tree_indices, medusa_position_ids, retrieve_indices.
    其中
    1. medusa_attention_mask用于控制解码时的可见范围
    2. medusa_tree_indices映射树节点到候选logits
    3. medusa_position_ids标识节点在树中的深度
    4. retrieve_indices回溯路径，用于验证和决策
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # 第一步：排序MEDUSA路径
    # Sort the medusa_choices based on their lengths and then their values
    # 先按照路径长度(树深度)排序，然后再按路径值排序
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    # 计算总的节点数，+1是为了包含跟节点
    medusa_len = len(sorted_medusa_choices) + 1

    # 第二步：统计各深度层级的节点数量
    # Initialize depth_counts to keep track of how many choices have a particular depth
    # depth_counts用于记录不同深度的节点数量， prev_depth为最大深度。用于后续构造注意力掩码与位置ID
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)  # 新深度层级
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # 第三步：为MEDUSA创建attention掩码。设计原理为：1. 所有节点关注根节点(第0列全1)；2.节点关注自身(对角线为1)；3.节点关注其所有祖先节点(路径前缀)
    # Create the attention mask for Medusa
    # 初始化一个medusa_len*medusa_len的单位阵,作为注意力掩码矩阵
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    # 将第一列的所有值设置为1。所有节点关注根节点
    medusa_attn_mask[:, 0] = 1
    # 初始化遍历的起始值
    start = 0
    for i in range(len(depth_counts)):  # 逐深度遍历
        for j in range(depth_counts[i]):  # 遍历每个深度的值
            # 从排序的medusa choice值中获取当前的medusa choice值
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue  # 第一层节点无需额外处理
            ancestor_idx = []
            # 获取每个深度不同节点的祖先节点在sorted_medusa_choices中的索引,并存储到ancestor_idx中
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(
                    cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1  # 将对应的节点的值改为1
        start += depth_counts[i]

    # 为medusa结构构造树索引。节点索引 = 路径末token ID + TOPK * 深度 + 1。TOPK为全局变量，表示每层生成的候选token数量。
    # Generate tree indices for the Medusa structure
    # 生成一个全零向量medusa_tree_indices，表示每个节点在树中代表的 token id
    # 将树节点映射到候选logits矩阵中的位置
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0  # 根节点索引=0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # 计算当前节点在候选logits中的位置
            medusa_tree_indices[start + j +
                                1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # 构造position ids(位置id)。用于区分不同深度的位置信息，根节点位置为0，第1层位置为1，第2层位置为2。
    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        # start+1为每个深度的起始索引，start+depth_counts[i]+1为每个深度结束索引
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]  # 倒序处理,避免路径重复
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(
                    sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length)
                        for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1  # 对所有值加1
    # 第1列全部设置为0
    retrieve_indices = torch.cat([torch.zeros(
        (retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers


def initialize_medusa(
        input_ids: torch.Tensor, 
        model: Union[MedusaModelLlama, MedusaModelMistral],
        medusa_attn_mask: torch.Tensor, 
        past_key_values: List[torch.Tensor])->Tuple[torch.Tensor, torch.Tensor]:    
    """
    Initializes the Medusa structure for a given model.
    初始化给定模型的Medusa结构

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids. 包含输入token ids的张量
    - model (MedusaLMHead): The model containing the Medusa layers and base model. 包含Medusa层与基础模型的MedusaLMHead的模型实例
    - medusa_attn_mask (torch.Tensor): The attention mask designed specifically for the Medusa structure. 专门为medusa结构设计的attention mask
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values. 包含历史隐状态与历史attention值的张量

    Returns:
    - medusa_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    # 执行首次前向传播。
    # medusa_logits: 来自 Medusa 头的预测(形状: [batch_size, num_heads, vocab_size])
    # outputs: 中间层输出
    # logits: 基础模型的原始预测（形状: [batch_size, seq_len, vocab_size]）
    medusa_logits, outputs, logits = model(
        input_ids,
        past_key_values=past_key_values,  # 提供历史key/value值
        output_orig=True,   # 要求返回基础模型的原始输出
        medusa_forward=True  # 激活medusa特定的前向传播
    )
    # 将medusa特定的注意力掩码注入基础模型
    model.base_model.model.medusa_mask = medusa_attn_mask 
    return medusa_logits, logits


def reset_medusa_mode(
    model: Union[MedusaModelLlama, MedusaModelMistral],
):
    """
    Resets the Medusa settings and the past key-values to their initial state.
    将medusa的设置与它们的历史keys-values重置为其初始状态

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model. 清空基础模型的medusa注意力掩码。
    2. Resets the Medusa mode in the base model. 重置基础模型的medusa模式
    3. Resets the current lengths in the past key-values to zero for all layers. 将所有层的历史key-values的当前长度cur_length参数重置为0

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]):
    """
    Resets the current lengths in the passed key-values to zero. 将历史key-values的当前长度重置为0

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def get_nucleus_one_token(
        logit: torch.Tensor,
        temperature: float,
        top_p: float):
    """
    Performs token sampling based on the nucleus (top-p) sampling method. 基于原子采样方法执行token采样

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.
    该函数使用原子采样策略从一个给定的logits分布选择一个token。该方法相比传统的top-k采样有更加可控的多样性的生产。

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC). B 是 batch size，C 是词表大小
        temperature (float): A temperature parameter to control the randomness in sampling. 
                             Higher values increase diversity, lower values make selections more deterministic.控制采样的“随机性”：越大越随机，越小越确定
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    # 如果 top_p >= 1，直接使用 softmax + multinomial 采样
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    # 温度缩放
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1) # 计算概率分布。形状保持 [batch_size, vocab_size]
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)# 对softmax处理后的概率进行排序
    cum_probs = torch.cumsum(sorted_logits, dim=-1)#计算累加概率分布
    # sorted_indices_to_remove为一个bool掩码，标记哪些 token 的累计概率超过了 top_p
    sorted_indices_to_remove = cum_probs > top_p
    # 这里将第一个 token 强制保留（即使它本身就已经超过 top_p），防止没有 token 可选
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # 右移掩码
    sorted_indices_to_remove[..., 0] = 0  # 第一列全部设置为False
    # 使用 scatter 把排序后的 mask 映射回原始 logit 的维度
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    # 把需要排除的 token 设为 -inf，这样 softmax 后概率会变成 0
    logit[indices_to_remove] = float('-inf')
    # 再次 softmax 得到新的概率分布（仅包含有效 token）；使用 multinomial 采样一个 token 返回
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens


def get_typical_one_token(
        logit: torch.Tensor,
        temperature: float,
        posterior_threshold: float,
        posterior_alpha: float):
    """
    Implements token sampling based on the typical sampling method. 基于典型采样方法实现token的采样方法。

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.
    该方法使用typical sampling策略从一个给定的logits分布选择一个token。

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.(形状: [batch_size, vocab_size]
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    # 通过温度参数调整 logits 分布。temperature = 0：退化为贪婪搜索；temperature < 1：锐化分布，增加确定性；temperature > 1：平滑分布，增加多样性。
    logit = logit / temperature
    # 将调整后的 logits 转换为概率分布。形状保持 [batch_size, vocab_size]
    probs = torch.softmax(logit, dim=-1)
    # 计算熵。H(p) = -\sum_{i} p_i \log p_i；1e-5：防止 log(0) 的数值稳定项
    entropy = -torch.sum(
        probs * torch.log(probs + 1e-5), dim=-1
    )
    # 构建了一个动态的概率下限，用来过滤掉不典型的 token。
    threshold = torch.minimum(
        # 固定阈值：posterior_threshold（用户设定的常量）
        torch.ones_like(entropy) * posterior_threshold,
        # 自适应阈值：$\exp(-H(p)) \times \alpha$
        torch.exp(-entropy) * posterior_alpha,
    )
    # 找出所有概率小于阈值的 token；
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    # 对筛选后的概率分布进行多项式采样
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens


def generate_candidates(
        medusa_logits: torch.Tensor,       # Medusa head输出的 logits [batch_size, num_heads, vocab_size]
        logits: torch.Tensor,              # 主模型输出的 logits (batch_size, seq_len, vocab_size)
        tree_indices: Union[torch.Tensor, List[torch.Tensor]],      # 树结构的索引列表，用于映射候选token到树状位置
        retrieve_indices: Union[torch.Tensor, List[torch.Tensor]],  # 用于提取笛卡尔候选的索引
        temperature: float = 0,            # 控制采样随机性的温度参数
        posterior_threshold: float = 0.3,  # 典型采样的概率阈值
        posterior_alpha: float = 0.09,     # 典型采样的熵缩放因子
        top_p: float = 0.8,                # nucleus sampling 的累积概率阈值
        sampling: str = 'typical',         # 采样方法 ('typical' 或 'nucleus')
        fast: bool = False):               # 是否启用快速确定性解码
    """
    Generate candidates based on provided logits and indices.
    基于提供的logits与索引，生产候选tokens
    
    Parameters:
    - medusa_logits (torch.Tensor): Logits from a specialized Medusa structure, aiding in candidate selection.
    - logits (torch.Tensor): Standard logits from a language model.
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.
    - temperature (float, optional): Controls the diversity of the sampling process. Defaults to 0.
    - posterior_threshold (float, optional): Threshold for typical sampling. Defaults to 0.3.
    - posterior_alpha (float, optional): Scaling factor for the entropy-based threshold in typical sampling. Defaults to 0.09.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """
    # 对LM head的输出进行采样。如果 temperature == 0 或者启用了 fast 模式，则使用 贪心解码（greedy decoding），直接选概率最高的 token。
    # 输出形状为 (1,)，表示一个 token。
    # Greedy decoding: Select the most probable candidate from the original logits.
    if temperature == 0 or fast:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        # 根据指定的采样方式（典型采样或 nucleus 采样）进行随机采样。
        if sampling == 'typical':  # typical sampling
            candidates_logit = get_typical_one_token(logits[:, -1], temperature, posterior_threshold, posterior_alpha).squeeze(0)
        elif sampling == 'nucleus':  # nucleus sampling
            candidates_logit = get_nucleus_one_token(logits[:, -1], temperature, top_p).squeeze(0)
        else:
            raise NotImplementedError
    # Extract the TOPK candidates from the medusa logits.
    # 对medusa logits在最后一个时间步（-1）、第一个 head（0）的结果执行TOPK采样，得到candidates的索引
    candidates_medusa_logits = torch.topk(medusa_logits[:, 0, -1], TOPK, dim = -1).indices

    # Combine the selected candidate from the original logits with the topk medusa logits.
    # 将主模型logits中采样的候选logits(candidates_logit)与top-k的medusa候选logits进行拼接(candidates_medusa_logits)
    # 输出形状：(1 + TOPK,)
    candidates = torch.cat([candidates_logit, candidates_medusa_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    # 从candidates中拿到树对应的节点
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    # 
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    # 使用 retrieve_indices 提取笛卡尔形式的候选 token
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    # 笛卡尔形式的候选 token，形状 (N,)
    # 树结构形式的候选 token，形状 (1, M)
    return cart_candidates, tree_candidates


def tree_decoding(
    model: Union[MedusaModelLlama, MedusaModelMistral],   # 使用的语言模型（通常是一个 nn.Module）
    tree_candidates: torch.Tensor,        # 树状结构的候选 token 序列(形状如: [1, M])
    past_key_values: torch.Tensor,        # 注意力机制的KV缓存，用于避免重复计算历史token
    medusa_position_ids: torch.Tensor,    # Medusa buffer中对应的 position IDs
    input_ids: torch.Tensor,              # 当前已有的输入序列 token IDs
    retrieve_indices: torch.Tensor,       # 用于从 logits 中提取特定位置的索引
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    该函数的主要功能为:
    1. 使用语言模型对一组“树结构”的候选token进行前向计算
    2. 
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """

    # input_ids.shape[1] 是当前输入序列的长度；
    # medusa_position_ids 是 Medusa 模型生成的位置 ID（通常是相对于当前输入的一个偏移量）；
    # 相加后得到的是这些候选 token 在整个上下文中的实际位置；
    # 示例：如果当前已有 10 个 token，Medusa 生成的位置 ID 是 [0, 1, 2]，那么加上 10 就变成了 [10, 11, 12]。
    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    position_ids = medusa_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
    tree_medusa_logits, outputs, tree_logits = model(
        tree_candidates,  # 表示要输出原始模型的 logits；
        output_orig=True,  # 表示要输出原始模型的 logits；
        past_key_values=past_key_values,  # 提供缓存的状态（key/value pairs）以加速推理；
        position_ids=position_ids,  # 提供缓存的状态（key/value pairs）以加速推理；
        medusa_forward=True,  # 表示这是 Medusa 解码模式；
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]
    return medusa_logits, logits, outputs


def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples * n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask


def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = True
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
        
    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha, fast)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError

def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]
    medusa_logits = medusa_logits[
        :, None, best_candidate, accept_length : accept_length + 1
    ]
    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, medusa_logits, new_token


if __name__ == "__main__":
    from medusa_choices import mc_sim_7b_63
    # breakpoint()
    generate_medusa_buffers(medusa_choices=mc_sim_7b_63)