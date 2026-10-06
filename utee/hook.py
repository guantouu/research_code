import os
import torch.nn as nn
import numpy as np
import torch
import math
import csv
from modules.conv import QuantBnConv2d
from modules.quantizer import weight_to_strips

# NeuroSIM binary written into trace_command.sh, relative to the repo root (build it with setup_runpod.sh)
NEUROSIM_MAIN = './NeuroSim/Inference_pytorch/NeuroSIM/main'

def Neural_Sim(self, input, output):
    """
    Export one QuantBnConv2d as one NeuroSIM layer with strip-wise mixed precision.

    Each crossbar strip weight[o, :, kh, kw] is a column of I rows (I = input channels), so a layer
    is declared to NeuroSIM as a 1x1 conv with I rows and one column group per strip. NeuroSIM runs
    with a single synapse precision (wl_weight, the low bitwidth), so a strip of b bits occupies
    b / wl_weight column groups holding its offset-binary digits, most significant first: an 8-bit
    strip is two 4-bit column groups. Uniform 4-bit / 8-bit baselines use the same mapping.
    The input vectors are the activations at the kernel center, i.e. the input sampled with the
    layer stride; the other kernel positions read shifted copies of the same feature map.
    With strip_bits set (hardware_evaluation), only the strips of those bitwidths are exported, and a
    layer with none of them is left out of the trace (layers.txt lists the exported layers in order).
    """
    global model_n
    global wl_weight
    global wl_input
    global layer_info
    global strip_bits
    global layer_names

    strips = weight_to_strips(self.weight_integer).cpu().data.numpy()
    selected = [(strip, int(bit)) for strip, bit in zip(strips, self.weight_bit)
                if strip_bits is None or int(bit) in strip_bits]
    if not selected:
        return
    layer_names.append(str(self.name))

    input_file_name =  './layer_record_' + str(model_n) + '/input_' + str(self.name) + '.csv'
    weight_file_name =  './layer_record_' + str(model_n) + '/weight_' + str(self.name) + '.csv'
    with open('./layer_record_' + str(model_n) + '/trace_command.sh', "a") as f:
        f.write(weight_file_name+' '+input_file_name+' ')

    columns = []
    for strip, bit in selected:
        columns.extend(strip_to_columns(strip, bit, wl_weight))
    weight_matrix = np.stack(columns, axis=1)   # [I, #column groups]
    # the values are multiples of 2^(1-wl_weight), which %.8g writes exactly and compactly
    np.savetxt(weight_file_name, weight_matrix, delimiter=",", fmt='%.8g')

    stride = self.conv.stride
    input_x = input[0].cpu().data.numpy()
    input_size = input_x.shape
    layer_info.append([input_size[2], input_size[3], input_size[1], 1, 1, weight_matrix.shape[1], 0, stride[0]])

    input_x = input_x / max(np.abs(input_x).max(), 1e-12) * (1 - 2.0 ** (1 - wl_input))   # into [-1, 1) for dec2bin
    tensor = stretch_input(input_x, 1, (0, 0), stride)
    write_matrix_activation_conv(tensor, None, wl_input, input_file_name)

def strip_to_columns(strip, bit, column_bit):
    """
    Split one strip of signed integers into bit / column_bit NeuroSIM columns in [-1, 1].
    NeuroSIM maps a weight f to the unsigned integer 2^(column_bit-1) * (f + 1), so each column
    holds one column_bit-wide digit d of the offset-binary weight as f = d / 2^(column_bit-1) - 1.
    """
    if bit % column_bit != 0:
        raise ValueError("Strip bitwidth {} is not a multiple of the column bitwidth {}".format(bit, column_bit))
    unsigned = np.rint(strip).astype(np.int64) + 2 ** (bit - 1)
    columns = []
    for k in reversed(range(bit // column_bit)):
        digit = (unsigned >> (k * column_bit)) & (2 ** column_bit - 1)
        columns.append(digit / 2 ** (column_bit - 1) - 1)
    return columns

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
    with open('./layer_record_'+str(model_n)+'/layers.txt', 'w') as file:
        file.write('\n'.join(layer_names) + '\n')

    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model, wl_weight_, wl_activation, subArray, parallelRead, model_name, strip_bits_=None):
    """
    wl_weight_ is the NeuroSIM synapse precision, i.e. the bitwidth of one column group (the low
    bitwidth of the strip bit config); higher-bitwidth strips use several column groups.
    strip_bits_ (e.g. {8}) exports only the strips of those bitwidths (one array of a dual-crossbar chip,
    see dual_crossbar_eval.py); None exports every strip.
    """
    global model_n
    global wl_weight
    global wl_input
    global layer_info
    global strip_bits
    global layer_names
    model_n = model_name
    wl_weight = wl_weight_
    wl_input = wl_activation
    layer_info = []
    strip_bits = None if strip_bits_ is None else {int(b) for b in strip_bits_}
    layer_names = []

    hook_handle_list = []
    if not os.path.exists('./layer_record_'+str(model_name)):
        os.makedirs('./layer_record_'+str(model_name))
    if os.path.exists('./layer_record_'+str(model_name)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(model_name)+'/trace_command.sh')
    f = open('./layer_record_'+str(model_name)+'/trace_command.sh', "w")
    f.write(NEUROSIM_MAIN+' ./layer_record_'+str(model_name)+'/NetWork.csv '+str(wl_weight)+' '+str(wl_activation)+' '+str(subArray)+' '+str(parallelRead)+' ')
    f.close()

    for name, layer in model.named_modules():
        if isinstance(layer, QuantBnConv2d):
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list
