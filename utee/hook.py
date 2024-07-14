import os
import torch.nn as nn
import numpy as np
import torch
import math
import csv
from modules.conv import QuantBnConv2d
bit_type = '4_bit'

def Neural_Sim(self, input, output): 
    global model_n
    global wl_input
    global layer_info

    input_file_name =  './layer_record_' + str(model_n) + '/input_' + str(self.name) + '.csv'
    weight_file_name =  './layer_record_' + str(model_n) + '/weight_' + str(self.name) + '.csv'
    f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
    f.write(weight_file_name+' '+input_file_name+' ')

    k_size = self.conv.kernel_size[0]
    weight_bit = self.weight_bit
    weight_q = self.weight_integer
    in_channels = self.conv.in_channels
    out_channels = self.conv.out_channels

    bit_8_matrix = []
    bit_4_matrix = []

    
    if bit_type == '4_bit' or bit_type == '8_bit':
        input_reshape = weight_q.cpu().data.numpy().reshape(-1, in_channels)
        for i, bit in enumerate(weight_bit):
            if bit == 4:
                bit_4_matrix.append(input_reshape[i])
            else:
                bit_8_matrix.append(input_reshape[i])
        
        bit_8_matrix = process_matrix(bit_8_matrix, k_size, in_channels)
        bit_4_matrix = process_matrix(bit_4_matrix, k_size, in_channels)

        if bit_type == '8_bit':
            if len(bit_8_matrix) != 0:
                out_channels = bit_8_matrix.shape[0]
                write_matrix_weight(bit_8_matrix, weight_file_name)
            else:
                out_channels = 0
                np.savetxt(weight_file_name, np.array([0.0000]), delimiter=",",fmt='%10.5f') 
        if bit_type == '4_bit':
            if len(bit_4_matrix) != 0:
                out_channels = bit_4_matrix.shape[0]
                write_matrix_weight(bit_4_matrix, weight_file_name)
            else:
                out_channels = 0
                np.savetxt(weight_file_name, np.array([0.0000]), delimiter=",",fmt='%10.5f')
    else:
        input_reshape = weight_q.cpu().data.numpy()
        write_matrix_weight(input_reshape, weight_file_name)

    padding = self.conv.padding
    stride = self.conv.stride

    tensor = stretch_input(input[0].cpu().data.numpy(), k_size, padding, stride)
    write_matrix_activation_conv(tensor, None, wl_input, input_file_name)

    input_size = input[0].shape
    layer_info.append([input_size[2], input_size[3], input_size[1], k_size, k_size, out_channels, 0, stride[0]])

def write_matrix_weight(input_matrix, filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')

def process_matrix(matrix_list, k_size, in_channels):
    remainder = len(matrix_list) % (k_size * k_size)
    if remainder != 0:
        matrix_list = matrix_list[:-remainder]
    
    if len(matrix_list) != 0:
        matrix_reshaped = np.vstack(matrix_list).reshape(-1, in_channels, k_size, k_size)
        return matrix_reshaped
    else:
        return []


def write_matrix_activation_conv(input_matrix, fill_dimension, length,filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def stretch_input(input_matrix, window_size=5, padding=(0,0),stride=(1,1)):
    input_shape = input_matrix.shape
    output_shape_row = int((input_shape[2] + 2*padding[0] -window_size) / stride[0] + 1)
    output_shape_col = int((input_shape[3] + 2*padding[1] -window_size) / stride[1] + 1)
    item_num = int(output_shape_row * output_shape_col)
    output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
    iter = 0
    if (padding[0] != 0):
        input_tmp = np.zeros((input_shape[0], input_shape[1], input_shape[2] + padding[0]*2, input_shape[3] + padding[1] *2))
        input_tmp[:, :, padding[0]: -padding[0], padding[1]: -padding[1]] = input_matrix
        input_matrix = input_tmp
    for i in range(output_shape_row):
        for j in range(output_shape_col):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i*stride[0]:i*stride[0]+window_size,j*stride[1]:j*stride[1]+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def remove_hook_list(hook_handle_list):
    global layer_info
    global model_n

    filename = './layer_record_'+str(model_n)+'/NetWork.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(layer_info)
    
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model, wl_weight, wl_activation, subArray, parallelRead, model_name): 
    global model_n
    global wl_input
    global layer_info
    model_n = model_name
    wl_input = 8
    layer_info = []
    
    hook_handle_list = []
    if not os.path.exists('./layer_record_'+str(model_name)):
        os.makedirs('./layer_record_'+str(model_name))
    if os.path.exists('./layer_record_'+str(model_name)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(model_name)+'/trace_command.sh')
    f = open('./layer_record_'+str(model_name)+'/trace_command.sh', "w")
    f.write('./NeuroSIM/main ./layer_record_'+str(model_name)+'/NetWork.csv '+str(wl_weight)+' '+str(wl_activation)+' '+str(subArray)+' '+str(parallelRead)+' ')
    
    for name, layer in model.named_modules():
        if isinstance(layer, QuantBnConv2d):
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list

