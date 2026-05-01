import numpy as np
import scipy.io as sio
import torch

def load_all_data(info_dicts):
    """
    data_info = {
    "img_type": "MT",
    "path": './data/denoise_acq_patient.mat'
    "in"  : [(0,0), (1,4)],
    "out" : [(0,4), (1,0), (1,2)],
    }
    """
    result_in = []
    result_out = []
    for d in info_dicts:
        # if type == "MT":
        #     tmp = load_MT(d)
        #     result_in.append(tmp[0])
        #     result_out.append(tmp[1])
        # else:
            tmp = load_data(d)
            result_in.append(tmp[0])
            result_out.append(tmp[1])
    # return np.concatenate(result_in, axis=1), np.concatenate(result_out, axis=1)
    return torch.cat(result_in, dim=1), torch.cat(result_out, dim=1)

def load_data(d):

    in_channel = len(d["in"]) 
    out_channel = len(d["out"])
    
    mat_mt = sio.loadmat(d["path"])
    img = mat_mt['data']
    if d['img_type'] == 'MTMap':
        img = np.expand_dims(img, axis=-1)    
    input_list = []
    output_list = []
    mtmap_list = []
    
    N_subjects, H, W, N_slices, channel = img.shape
    
    # Extract the cell array for MT & T1
    for i in range(N_subjects):
        """
        iterate thru all images
            append all the slice with specific channel to input/output
            input (p,o) : 0,0 1,0 1,2 1,4
        """
        img_input = np.zeros((N_slices, H, W,in_channel))
        img_output = np.zeros((N_slices, H, W, out_channel))
        img_cur = img[i]
        input_ptr = 0
        output_ptr = 0
        for fa in range(channel):
            if fa in d['in']:
                img_input[:,:,:,input_ptr] = np.transpose(img_cur[:,:,:,fa], (2, 0, 1))
                input_ptr += 1
            elif fa in d['out']:
                img_output[:,:,:,output_ptr] = np.transpose(img_cur[:,:,:,fa], (2, 0, 1))  
                output_ptr += 1
                    
            
        input_list.append(img_input)
        output_list.append(img_output)

    # Concatenate the data along the first dimension (slice dimension)
    input_data = np.concatenate(input_list, axis=0)
    output_data = np.concatenate(output_list, axis=0)
    
    # Transpose (n, height,weiht, channels) --> (n, channel, weight, channels)
    input_data = torch.from_numpy(input_data).permute(0, 3, 1, 2).float()
    output_data = torch.from_numpy(output_data).permute(0, 3, 1, 2).float()
    print(d['img_type'], "Input/Output data shape:", input_data.shape, output_data.shape)

    return torch.tensor(input_data), torch.tensor(output_data)

def load_MT(d):
    in_channel = len(d["in"]) 
    out_channel = len(d["out"])
    
    mat_mt = sio.loadmat(d["path"])
    img = mat_mt['data']
    
    input_list = []
    output_list = []
    mtmap_list = []

    channel = img[0].shape[-1]
    array_shape = img[0].shape
    N_subjects = img.shape[0]
    N_slices = img.shape[-2]
    H, W = array_shape[0], array_shape[1]
    # Extract the cell array for MT & T1
    for i in range(N_subjects):
        """
        iterate thru all images
            append all the slice with specific channel to input/output
            input (p,o) : 0,0 1,0 1,2 1,4
        """
        img_input = np.zeros((N_slices, H, W,in_channel))
        img_output = np.zeros((N_slices, H, W, out_channel))
        img_cur = img[i]
        input_ptr = 0
        output_ptr = 0
        for fa in range(channel):
            if fa in d['in']:
                img_input[:,:,:,input_ptr] = np.transpose(img_cur[:,:,:,fa], (2, 0, 1))
                input_ptr += 1
            elif fa in d['out']:
                img_output[:,:,:,output_ptr] = np.transpose(img_cur[:,:,:,fa], (2, 0, 1))  
                output_ptr += 1
                    
            
        input_list.append(img_input)
        output_list.append(img_output)

    # Concatenate the data along the first dimension (slice dimension)
    input_data = np.concatenate(input_list, axis=0)
    output_data = np.concatenate(output_list, axis=0)
    
    # Transpose (n, height,weiht, channels) --> (n, channel, weight, channels)
    input_data = torch.from_numpy(input_data).permute(0, 3, 1, 2).float()
    output_data = torch.from_numpy(output_data).permute(0, 3, 1, 2).float()
    print(d['img_type'], "Input/Output data shape:", input_data.shape, output_data.shape)

    return torch.tensor(input_data), torch.tensor(output_data)

def train_test_split(input_data, output_data, test_input_data=None, test_output_data=None):
    data_size = input_data.shape[0]
    if test_input_data is None:
        data_size -= 10
    train_data = {}
    valid_data = {}
    test_data = {}
    train_idx = np.random.choice(data_size, size=int(0.9*data_size),replace=False)
    valid_idx = np.setdiff1d(np.arange(data_size), train_idx)
    
    train_data['in'] = input_data[train_idx,:]
    train_data['out'] = output_data[train_idx,:]
    print("Train shape input: ", train_data['in'].shape)
    print("Train shape output: ", train_data['out'].shape)
    
    valid_data['in'] = input_data[valid_idx,:]
    valid_data['out'] = output_data[valid_idx,:]

    test_data['in'] = test_input_data
    test_data['out'] = test_output_data
    if test_input_data is None:
        test_data['in'] = input_data[-10:,:]
        test_data['out'] = output_data[-10:,:]
    return train_data, valid_data, test_data


# stage 1 prediction
import os
from models import *
def stage1_prediction(data_input, pth_name, in_channels, out_channels):
    import os
    from torch.utils.data import DataLoader, TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the desired device (GPU or CPU)
    cwd = os.getcwd()

    model_path = os.path.join(
        cwd,
        r'checkpoints',
        f"{pth_name}"
    )
    print("Loading from path: ", model_path)
    state_dict = torch.load(model_path)
    model = unet(in_channels,out_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    batch_size = 32  # Adjust this according to your system's memory capacity
    input_data = data_input.float()  # Ensure the input data is in float format
    dataset = TensorDataset(input_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize a list to store predictions
    all_predictions = []
    
    # Iterate through the data in batches
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in dataloader:
            # Move the batch to the device
            batch_input = batch[0].to(device)
            
            # Predict and move the output to CPU
            batch_output = model(batch_input).detach().cpu().float()
            
            # Append the batch output to the list
            all_predictions.append(batch_output)
    
    # Concatenate all batch predictions into a single tensor
    return torch.cat(all_predictions, dim=0).cpu()
    # input_sample = input_data.cpu().squeeze(0)  # Remove batch dimension (if needed)