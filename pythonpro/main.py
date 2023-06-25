import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from func import *

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_word_list = WordList('./dataset/pos.txt')
neg_word_list = WordList('./dataset/neg.txt')
word_dict = Dictionary()

for index in range(pos_word_list.contentNum):
    pos_word_sentence = pos_word_list.get_word_list()
    word_dict.add_word_list(pos_word_sentence)

for index in range(neg_word_list.contentNum):
    neg_word_sentence = neg_word_list.get_word_list()
    word_dict.add_word_list(neg_word_sentence)

neg_word_list.index = 0
pos_word_list.index = 0

net = nn.RNN(len(word_dict.dictionary), 2000, 2, nonlinearity='relu').to(device)
h0 = torch.randn(2, 1, 2000).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Train
sequence_length_list = []
input_tensor_list = []
target_list = []
for index in range(pos_word_list.contentNum):
    pos_word_sentence = pos_word_list.get_word_list()
    sequence_length_list.append(len(pos_word_sentence))

    input_list = []
    for word_index in range(len(pos_word_sentence)):
        if pos_word_sentence[word_index] in word_dict.dictionary.keys():
            input_vector = np.zeros(len(word_dict.dictionary))
            input_vector[word_dict.dictionary[pos_word_sentence[word_index]]] = 1
            input_list.append(input_vector)

    input_tensor_list.append(input_list)  # 添加输入向量到列表，不转换为张量
    target_list.append(torch.tensor([0, 1], dtype=torch.float32))
# 将序列长度列表转换为循环外的张量
sequence_lengths = torch.tensor(sequence_length_list, dtype=torch.int64).to(device)
# 将输入张量列表转换为单个NumPy数组
input_tensor_array = np.array([np.array(seq) for seq in input_tensor_list], dtype=object)
# 找出最大序列长度
max_length = max(len(seq) for seq in input_tensor_list)
# 创建一个张量来存储填充序列
padded_input_tensor = torch.zeros(len(input_tensor_list), max_length, len(word_dict.dictionary)).to(device)
# 将输入张量数组转换为张量并填充序列
for i, seq in enumerate(input_tensor_array):
    seq_length = len(seq)
    padded_input_tensor[i, :seq_length, :] = torch.tensor(seq, dtype=torch.float32).to(device)
# 根据序列长度降序排序输入张量和序列长度
sorted_data = sorted(zip(padded_input_tensor, sequence_lengths, target_list), key=lambda x: x[1], reverse=True)
sorted_input_tensor, sorted_sequence_lengths, sorted_target_list = zip(*sorted_data)
# 创建批量输入张量、序列长度和目标张量
batch_input_tensor = torch.stack(sorted_input_tensor, dim=0).to(device)
sequence_lengths = torch.tensor(sorted_sequence_lengths, dtype=torch.int64).to(device)
target_tensor = torch.stack(sorted_target_list, dim=0).to(device)
# 为每个输入序列重置h0
h0 = torch.randn(2, len(sequence_lengths), 2000).to(device)
# 重塑h0以匹配批输入张量的形状
h0 = h0.expand(2, batch_input_tensor.size(0), h0.size(2))
# 打包输入序列
packed_input = torch.nn.utils.rnn.pack_padded_sequence(batch_input_tensor.unsqueeze(1), sequence_lengths, batch_first=True)
# 前向模式
output, hn = net(packed_input, h0)

loss_fn = nn.MSELoss()
loss = loss_fn(output[-1].squeeze(), target_tensor)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print('Training completed.')

# Test with a session
# 使用会话进行测试
print('输入一个会话（中文句子）：')
session = input()
inner_prog = re.compile(r'(\w.*?)[ ,.?]')
session_list = inner_prog.findall(session)

# 为会话准备输入张量
input_list = []
for word in session_list:
    if word in word_dict.dictionary.keys():
        input_vector = np.zeros(len(word_dict.dictionary))
        input_vector[word_dict.dictionary[word]] = 1
        input_list.append(input_vector)

input_tensor = torch.tensor(input_list, dtype=torch.float32).to(device)
h0 = torch.randn(2, 1, 2000).to(device)  # Reset hidden state for session

# 重塑h0以匹配输入张量的形状
h0 = h0.expand(2, input_tensor.size(0), h0.size(2))

output, hn = net(input_tensor.unsqueeze(1), h0)

print('输出:', output[-1].squeeze().to("cpu"))
