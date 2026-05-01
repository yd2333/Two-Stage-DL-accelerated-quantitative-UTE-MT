""" Visualize with Matplotlib """
import os
import numpy as np
import matplotlib.pyplot as plt


"""
Loss and Index
"""
# from keras import backend as K

# def ssim(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     mean_true = K.mean(y_true_f)
#     mean_pred = K.mean(y_pred_f)
#     std_true = K.std(y_true_f)
#     std_pred = K.std(y_pred_f)
    
#     k1 = 0.01
#     k2 = 0.03
    
#     C1 = K.pow(k1,2)
#     C2 = K.pow(k2,2)
    
#     # retval = (2*mean_true*mean_true+C1)*(2*std_true*std_pred+C2)/(K.pow(mean_true,2)+K.pow(mean_pred,2)+C1)/(K.pow(std_true,2)+K.pow(std_pred,2)+C2)
#     numerator = (2 * mean_true * mean_pred + C1) * (2 * std_true * std_pred + C2)
#     denominator = (K.pow(mean_true, 2) + K.pow(mean_pred, 2) + C1) * (K.pow(std_true, 2) + K.pow(std_pred, 2) + C2)
    
#     retval = numerator / denominator
#     return retval
    
# def l1dist(x,y):
#         return K.sum(K.abs(x-y))    
    
# def ssim_loss(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     l1norm = l1dist(y_true_f, y_pred_f)
#     return l1norm-5000*ssim(y_true, y_pred)


# def save_png(gt, prediction, image, save_path, MTmap):
#     """
#     gt : ground truth values
#     prediction : output of the network
#     image : original MRI image
#     save_dir : directory to save the image
#     All inputs have size: 160x160x1
#     """

#     if MTmap:
#         # Normalize and prepare images
#         gt_norm = gt * 1000
#         prediction_norm = prediction * 1000
#         diff = np.abs(gt_norm - prediction_norm) # *5
    
#         # Calculate vmin and vmax for each data array
#         gt_vmin, gt_vmax = gt_norm.min(), gt_norm.max()
#     else:
#         gt_norm = gt
#         prediction_norm = prediction
#         diff = np.abs(gt_norm - prediction_norm) # *5
#         gt_vmin, gt_vmax = gt_norm.min(), gt_norm.max()
#         image = np.zeros(gt.shape)
    
#     # Create a figure with 4 subplots in a row
#     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
#     # Plot each image separately
#     axes[0].imshow(gt_norm[:, :, 0], cmap='gray', vmin=gt_vmin, vmax=gt_vmax)
#     axes[0].set_title('Ground Truth')
#     axes[0].axis('off')
    
#     axes[1].imshow(prediction_norm[:, :, 0], cmap='gray', vmin=gt_vmin, vmax=gt_vmax)
#     axes[1].set_title('Prediction')
#     axes[1].axis('off')
    
#     axes[2].imshow(diff[:, :, 0], cmap='gray', vmin=-100, vmax=100)
#     axes[2].set_title('Difference')
#     axes[2].axis('off')
    
#     axes[3].imshow(image[:, :, 0], cmap='gray', vmin=-10, vmax=3468)
#     axes[3].set_title('Original Image')
#     axes[3].axis('off')
    
#     # Save the figure
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# for visulization
def display_pytorch(gt, net_out, save_path):
    
    # 4, 160, 160
    # Normalize the data if needed
    gt_norm = gt  # Use slice to remove the first dimension
    prediction_norm = net_out  # Same for net_out

    # Define min/max for visualization
    gt_vmin, gt_vmax = gt_norm[:8].min(), gt_norm[:8].max() # You can adjust based on your actual data ranges

    # Set up the figure
    channel = gt.shape[0]
    fig, axs = plt.subplots(channel, 4, figsize=(16, 24))  # 4 rows for each channel, 3 columns for GT, net_out, and diff
    i = 0
    

    if channel == 1:
        diff = np.abs(gt_norm[i] - prediction_norm[i])
        
        # Display ground truth
        axs[ 0].imshow(gt_norm[i], cmap='gray', vmin=gt_vmin, vmax=gt_vmax)
        axs[ 0].set_title(f'Ground Truth (Channel {i+1})')
        axs[ 0].axis('off')

        # Display network output
        axs[ 1].imshow(prediction_norm[i], cmap='gray', vmin=gt_vmin, vmax=gt_vmax)
        axs[ 1].set_title(f'Network Output (Channel {i+1})')
        axs[ 1].axis('off')

        # Display the difference
        axs[2].imshow(diff*10, cmap='gray',vmin=gt_vmin, vmax=gt_vmax)
        axs[2].set_title(f'Difference *10 (Channel {i+1}): {diff.mean()}')
        axs[2].axis('off')
    else:
        for i in range(channel):
            # Calculate the absolute difference
            diff = np.abs(gt_norm[i] - prediction_norm[i])
            
            # Display ground truth
            axs[i, 0].imshow(gt_norm[i], cmap='gray', vmin=gt_vmin, vmax=gt_vmax)
            axs[i, 0].set_title(f'Ground Truth (Channel {i+1})')
            axs[i, 0].axis('off')
    
            # Display network output
            axs[i, 1].imshow(prediction_norm[i], cmap='gray', vmin=gt_vmin, vmax=gt_vmax)
            axs[i, 1].set_title(f'Network Output (Channel {i+1})')
            axs[i, 1].axis('off')

            
            # Display the difference
            axs[i, 2].imshow(diff, cmap='gray',vmin=gt_vmin, vmax=gt_vmax)
            axs[i, 2].set_title(f'Difference *10 (Channel {i+1}): {diff.mean()}')
            axs[i, 2].axis('off')

             # Display the difference by 10
            axs[i, 3].imshow(diff*10, cmap='gray',vmin=gt_vmin, vmax=gt_vmax)
            axs[i, 3].set_title(f'Difference *10 (Channel {i+1}): {diff.mean()}')
            axs[i, 3].axis('off')   
   
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close()