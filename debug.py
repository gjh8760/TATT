import torch
import ipdb

device = torch.device('cuda:0')

# tensor_256_0 = torch.load('256/0_LR.pt', device)
# # tensor_256_1 = torch.load('256/1_TP.pt', device)
# # tensor_256_2 = torch.load('256/2_after_TPS.pt', device)
# # tensor_256_3 = torch.load('256/3_after_9x9_conv.pt', device)
# # tensor_256_4 = torch.load('256/4_TP_map.pt', device)
# # tensor_256_5 = torch.load('256/5_SR_img.pt', device)
# #
# # tensor_512_0 = torch.load('512/0_LR.pt', device)[:256]
# # tensor_512_1 = torch.load('512/1_TP.pt', device)[:256]
# # tensor_512_2 = torch.load('512/2_after_TPS.pt', device)[:256]
# # tensor_512_3 = torch.load('512/3_after_9x9_conv.pt', device)[:256]
# # tensor_512_4 = torch.load('512/4_TP_map.pt', device)[:256]
# # tensor_512_5 = torch.load('512/5_SR_img.pt', device)[:256]
# #
# # diff_0 = float(((tensor_256_0 - tensor_512_0) ** 2).mean())
# # diff_1 = float(((tensor_256_1 - tensor_512_1) ** 2).mean())
# # diff_2 = float(((tensor_256_2 - tensor_512_2) ** 2).mean())
# # diff_3 = float(((tensor_256_3 - tensor_512_3) ** 2).mean())
# # diff_4 = float(((tensor_256_4 - tensor_512_4) ** 2).mean())
# # diff_5 = float(((tensor_256_5 - tensor_512_5) ** 2).mean())
# #
# # print('diff_0: ', diff_0)
# # print('diff_1: ', diff_1)
# # print('diff_2: ', diff_2)
# # print('diff_3: ', diff_3)
# # print('diff_4: ', diff_4)
# # print('diff_5: ', diff_5)

# tensor_256_0 = torch.load('256_crnn/0_after_conv.pt', device)
# tensor_256_0_5 = torch.load('256_crnn/0.5_before_rnn.pt', device)
# tensor_256_1 = torch.load('256_crnn/1_after_rnn.pt', device)
#
# tensor_512_0 = torch.load('512_crnn/0_after_conv.pt', device)[:256]
# tensor_512_0_5 = torch.load('512_crnn/0.5_before_rnn.pt', device)[:, :256, :]
# tensor_512_1 = torch.load('512_crnn/1_after_rnn.pt', device)[:, :256, :]
#
# diff_0 = float(((tensor_256_0 - tensor_512_0) ** 2).mean())
# diff_0_5 = float(((tensor_256_0_5 - tensor_512_0_5) ** 2).mean())
# diff_1 = float(((tensor_256_1 - tensor_512_1) ** 2).mean())
#
# print('diff_0: ', diff_0)
# print('diff_0.5: ', diff_0_5)
# print('diff_1: ', diff_1)

tensor_256_0 = torch.load('256_lstm/0_input.pt', device)
tensor_256_1 = torch.load('256_lstm/1_after_lstm.pt', device)
tensor_256_2 = torch.load('256_lstm/2_after_view.pt', device)
tensor_256_3 = torch.load('256_lstm/3_after_linear.pt', device)
tensor_256_4 = torch.load('256_lstm/4_after_view.pt', device)

tensor_256_0_d = torch.load('256_lstm/0_input.pt', device).double()
tensor_256_1_d = torch.load('256_lstm/1_after_lstm.pt', device).double()
tensor_256_2_d = torch.load('256_lstm/2_after_view.pt', device).double()
tensor_256_3_d = torch.load('256_lstm/3_after_linear.pt', device).double()
tensor_256_4_d = torch.load('256_lstm/4_after_view.pt', device).double()

tensor_512_0 = torch.load('512_lstm/0_input.pt', device)#[:, :256, :]   # (26, B, 256)
tensor_512_1 = torch.load('512_lstm/1_after_lstm.pt', device)#[:, :256, :]  # (26, B, 512)
tensor_512_2 = torch.load('512_lstm/2_after_view.pt', device)#[:, :]  # (B*26, 512)
# 이 사이에 nn.linear 통과
tensor_512_3 = torch.load('512_lstm/3_after_linear.pt', device)#[:6656, :]  # (26*B, 37)
tensor_512_4 = torch.load('512_lstm/4_after_view.pt', device)#[:, :256, :]  # (26, B, 37)

tensor_512_0_d = torch.load('512_lstm/0_input.pt', device).double()#[:, :256, :]   # (26, B, 256)
tensor_512_1_d = torch.load('512_lstm/1_after_lstm.pt', device).double()#[:, :256, :]  # (26, B, 512)
tensor_512_2_d = torch.load('512_lstm/2_after_view.pt', device).double()#[:, :]  # (B*26, 512)
# 이 사이에 nn.linear 통과
tensor_512_3_d = torch.load('512_lstm/3_after_linear.pt', device).double()#[:6656, :]  # (26*B, 37)
tensor_512_4_d = torch.load('512_lstm/4_after_view.pt', device).double()#[:, :256, :]  # (26, B, 37)


weight_256 = torch.load('weights/weight_256.pt', device)    # CRNN 두번째 BiLSTM 모듈의 linear layer weight, (37, 512)
bias_256 = torch.load('weights/bias_256.pt', device)        # (37)
weight_512 = torch.load('weights/weight_512.pt', device)
bias_512 = torch.load('weights/bias_512.pt', device)

weight_256_d = torch.load('weights/weight_256.pt', device).double()   # CRNN 두번째 BiLSTM 모듈의 linear layer weight, (37, 512)
bias_256_d = torch.load('weights/bias_256.pt', device).double()        # (37)
weight_512_d = torch.load('weights/weight_512.pt', device).double()
bias_512_d = torch.load('weights/bias_512.pt', device).double()

ipdb.set_trace()

# before nn.linear
for n in range(tensor_256_2.shape[0]):
    print(((tensor_512_2[512*(n//256)+(n%256)] - tensor_256_2[n])**2).sum())

# after nn.linear
for n in range(tensor_256_3.shape[0]):
    print(((tensor_512_3[512 * (n // 256) + (n % 256)] - tensor_256_3[n]) ** 2).sum())

tensor_256_3_new = torch.matmul(tensor_256_2, weight_256.T) + bias_256
tensor_512_3_new = torch.matmul(tensor_512_2, weight_512.T) + bias_512

# nn.linear에 문제 없는지 체크.
print(tensor_256_3 - tensor_256_3_new)
print(tensor_512_3 - tensor_512_3_new)

# 1. input matrix들 같은지 체크
print('-tensor-')
for n in range(tensor_256_2.shape[0]):
    print(((tensor_512_2[512*(n//256)+(n%256)] - tensor_256_2[n])**2).sum())

print('-weight-')
print(weight_256 - weight_512)

print('-bias-')
print(bias_256 - bias_512)

# 2. output matrix 같은지 체크
tensor_256_3 = tensor_256_3
tensor_256_3_new = torch.matmul(tensor_256_2, weight_256.T) + bias_256
tensor_256_3_new_0 = torch.matmul(tensor_256_2[0:1], weight_256.T) + bias_256
tensor_256_3_new_1 = torch.matmul(tensor_256_2[0:10], weight_256.T) + bias_256
tensor_256_3_new_2 = torch.matmul(tensor_256_2[0:100], weight_256.T) + bias_256

tensor_512_3 = tensor_512_3
tensor_512_3_new = torch.matmul(tensor_512_2, weight_512.T) + bias_512
tensor_512_3_new_0 = torch.matmul(tensor_512_2[0:1], weight_512.T) + bias_512
tensor_512_3_new_1 = torch.matmul(tensor_512_2[0:10], weight_512.T) + bias_512
tensor_512_3_new_2 = torch.matmul(tensor_512_2[0:100], weight_512.T) + bias_512


print(tensor_256_3[0] - tensor_512_3[0])
print(tensor_256_3_new[0] - tensor_512_3[0])
print(tensor_256_3_new_0[0] - tensor_512_3[0])
print(tensor_256_3_new_1[0] - tensor_512_3[0])
print(tensor_256_3_new_2[0] - tensor_512_3[0])

print(tensor_256_3[0] - tensor_512_3[0])
print(tensor_256_3[0] - tensor_512_3_new[0])
print(tensor_256_3[0] - tensor_512_3_new_0[0])
print(tensor_256_3[0] - tensor_512_3_new_1[0])
print(tensor_256_3[0] - tensor_512_3_new_2[0])

print(tensor_256_3_new_0 - tensor_512_3_new_0)

# .view 없이 nn.linear
tensor_256_new = torch.matmul(tensor_256_1, weight_256.T) + bias_256
tensor_512_new = torch.matmul(tensor_512_1, weight_512.T) + bias_512

# double
tensor_256_3_new_d = torch.matmul(tensor_256_2_d, weight_256_d.T) + bias_256_d
tensor_512_3_new_d = torch.matmul(tensor_512_2_d, weight_512_d.T) + bias_512_d
tensor_256_3_new_f = tensor_256_3_new_d.float()
tensor_512_3_new_f = tensor_512_3_new_d.float()
for n in range(tensor_256_3_new_d.shape[0]):
    print(((tensor_512_3_new_d[512*(n//256)+(n%256)] - tensor_256_3_new_d[n])**2).sum())
for n in range(tensor_256_3_new_f.shape[0]):
    print(((tensor_512_3_new_f[512*(n//256)+(n%256)] - tensor_256_3_new_f[n])**2).sum())


# diff_0 = float(((tensor_256_0 - tensor_512_0) ** 2).mean())
# diff_1 = float(((tensor_256_1 - tensor_512_1) ** 2).mean())
# diff_2 = float(((tensor_256_2 - tensor_512_2) ** 2).mean())
# diff_3 = float(((tensor_256_3 - tensor_512_3) ** 2).mean())
# diff_4 = float(((tensor_256_4 - tensor_512_4) ** 2).mean())

# print('crnn output shape: ', tensor_256_4.shape, tensor_512_4.shape)

# print('diff_0: ', diff_0)
# print('diff_1: ', diff_1)
# print('diff_2: ', diff_2)
# print('diff_3: ', diff_3)
# print('diff_4: ', diff_4)
