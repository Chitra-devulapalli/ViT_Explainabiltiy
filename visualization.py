from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from collections import OrderedDict, defaultdict

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP , Baselines
import os 
import torchvision.datasets as datasets
from torch.utils.data import dataset, DataLoader
from tqdm import tqdm


# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    "This function is from the original implementation, nothing to change here."
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

'''
change paths and create a new classlabel.txt file 
'''
val_dir = os.path.join('/scratch/eecs542f24_class_root/eecs542f24_class/shared_data/imagenet-100', 'val')
dir = os.path.join('/scratch/eecs542f24_class_root/eecs542f24_class/chitrakd/homework1', 'samples')
dataset = datasets.ImageFolder(
        dir,
        transform)
orig_dataset = datasets.ImageFolder(
        val_dir,
        transform)

correspondences = dataset.class_to_idx

with open('newclasslabels.txt') as f:
    d = dict(x.rstrip().split(None, 1) for x in f)

correspondences = { v:d.get(k, k) for k, v in correspondences.items() }

dataloader = DataLoader(dataset, batch_size=3, num_workers=0, drop_last=False, shuffle=False)
orig_dataloader = DataLoader(orig_dataset, batch_size=3, num_workers=0, drop_last=False, shuffle=False)
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad(): ## we don't do this because want to register grads for the visualization implementation
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(model, dataloader): 
    model.eval()
    # with torch.no_grad(): ## we don't do this because want to register grads for the visualization implementation
    acc1_total = 0
    acc5_total = 0
    total_items = 0
    for step, (x, y) in tqdm(enumerate(dataloader)):
        outputs = model(x.cuda())
        acc1, acc5 = accuracy(outputs, y.cuda(), topk=(1, 5))
        
        acc1_total += acc1*len(y)
        acc5_total += acc5*len(y)
        total_items += len(y)

    print(f"VALIDATION | Top 1 Acc: {acc1_total/total_items}, Top 5 Acc: {acc5_total/total_items}")
    

model = vit_LRP(pretrained=False, num_classes=100).cuda()
state_dict = torch.load('Model_B_weights.pth')

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')  # remove `module.` prefix in case it exists (when was trained with DDP it does)
    new_state_dict[name] = v


model.load_state_dict(new_state_dict, strict=True)
print('WEIGHTS LOADED')

# validate(model, dataloader)
# exit()


model.eval()
attribution_generator = LRP(model)
baseline_generator = Baselines(model)

def generate_visualization(original_image, Analyzer=None, class_index=None):

    if (Analyzer == "transformer_attribution"):
        " This function is from the original implementation to get Hila Chefer's method."
        transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method=Analyzer, index=class_index).detach()
    else:
        transformer_attribution = baseline_generator.generate_rollout(original_image.unsqueeze(0).cuda(), start_layer=0)

    print(transformer_attribution.shape)
    transformer_attribution = transformer_attribution.squeeze()
    # transformer_attribution = transformer_attribution[6:]
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    # transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, size=(14, 14), mode='bilinear')
    # print(transformer_attribution.shape)method
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, **kwargs):    
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = [idx for idx in predictions.data.topk(5, dim=1)[1][0].tolist() if idx in correspondences]
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(correspondences[cls_idx])
        if len(correspondences[cls_idx]) > max_str_len:
            max_str_len = len(correspondences[cls_idx])
    
    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, correspondences[cls_idx])
        output_string += ' ' * (max_str_len - len(correspondences[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


## THIS IS A LOOP OVER THE VALIDATION DATASET, WITH SOME SCAFFOLD FOR THE VISUALZIATION.
## YOU WILL NEED TO FIND ONE EXAMPLE OF SOMETHING MISCLASSIFIED AND ALSO LOOK AT THE 2 IMAGES (IN samples)
## YOU WILL ALSO NEED TO ADD CODE FOR THE REMAINING VISUALIZATIONS 
## I.E, THE CODE IS PROVIDED FOR THE NEW METHOD, YOU NEED MAKE FOR ROLLOUT
# TODO: MAKE A FUNCTION FOR ROLLOUT, CHECK RESULTS ON GIVEN IMAGES (see samples folder), FIND MISCLASSIFIED IMG FROM VAL SET, CHECK VISUALZIATION THERE TOO; DO THIS FOR BOTH MODELS 
# TODO: (MODIFY MODEL W REGISTERS TO EXTACT TOKEN VALUES (different file)), ADD VISUALIZATION FOR TOKEN NORMS (L2) AT DIFFERENT LAYERS (1, 6, 12) AVERAGED OVER HEADS, FOR THE 3 IMAGES
# TODO: FOR ALL IMAGES IN THE VALIDATION SET, RANK THE NORMS OF THEIR VALIDATION TOKENS AT DIFFERENT LAYERS, PLOT THAT AS A NORMALIZED HISTOGRAM.
# FEEL FREE TO USE ANY FORMAT FOR THIS FILE AND MAKE HELPER FILES IF DESIRED. THE ONLY GOAL IS THAT YOUR CODE PRODUCES THE RIGHT GRAPHS.
# HARDCODING IS OK, COMMENTING OUT PARTS THEN UNCOMMENTING THEM AGAIN IS OK.



fig, axs = plt.subplots(3, 4)

for step, (i, y) in enumerate(dataloader):

    for idx in range(i.size(0)):  # Loop through images in the batch (3 images in this case)
        print(f"Class: {y[idx]}")
        print(correspondences[int(y[idx].item())])

        # Permute image tensor to match the format (H, W, C)
        image = torch.permute(i[idx], (1, 2, 0))
        dog_cat_image = image.detach().cpu().numpy()

        # Display the original image in the first column of the current row (idx)
        axs[idx, 0].imshow(dog_cat_image)
        axs[idx, 0].axis('off')  # Turn off the axis for a cleaner display

        #TRUE if you want norms
        return_token_norms = True
        # Get model output (e.g., classifier output or similar)
        result = model(i.cuda(),return_token_norms=return_token_norms)

        if return_token_norms:
            output, token_norms = result

            for im, layer in enumerate([1, 6, 12]):  # im is used for column index
                norms = token_norms[f'layer_{layer}']
                
                # Visualize for the current image (idx) in the batch
                norm_min = norms[idx].min()  # Use norms for this image
                norm_max = norms[idx].max()
                normalized = (norms[idx] - norm_min) / (norm_max - norm_min)

                # Reshape to 14x14 for visualization (assuming 196 patch tokens)
                norm_heatmap = normalized.view(14, 14).cpu().detach().numpy()

                # Visualize in the corresponding column for each layer
                axs[idx, im + 1].imshow(norm_heatmap, cmap='viridis')  # im + 1 to skip the original image column
                axs[idx, im + 1].axis('off')

            # axs[idx, 2].imshow(dog)
            # axs[idx, 2].axis('off')

            # axs[idx, 3].imshow(cat)
            # axs[idx, 3].axis('off')


        else: 
            output = result


            top2_predictions = torch.topk(output[idx], 2)  # Get the top 2 predicted class indices for the current image
            second_most_confident_class = top2_predictions.indices[1].item()  # Get the second class index
        
            dog_ro = generate_visualization(i[idx], Analyzer="rollout")
            dog = generate_visualization(i[idx], Analyzer="transformer_attribution")
            cat = generate_visualization(i[idx], class_index=second_most_confident_class, Analyzer="transformer_attribution")  # Optional
            # Display visualizations in the remaining columns of the current row (idx)
            axs[idx, 1].imshow(dog_ro)
            axs[idx, 1].axis('off')

            axs[idx, 2].imshow(dog)
            axs[idx, 2].axis('off')

            axs[idx, 3].imshow(cat)
            axs[idx, 3].axis('off')
 
    # print('Class: ', y[step])
    # print(correspondences[int(y[step].item())])

    # image = torch.permute(i[0], (1,2,0))
    
    # dog_cat_image = image.detach().cpu().numpy()
    
    # # fig, axs = plt.subplots(3, 4)
    # axs[step, 0].imshow(dog_cat_image);
    # axs[step, 0].axis('off');

    # output = model(i.cuda())


    # # print_top_classes(output)

    # dog = generate_visualization(i[0], Analyzer = "transformer_attribution")
    # dog_ro = generate_visualization(i[0], Analyzer = "rollout")

    # cat = generate_visualization(i[0], class_index=2, Analyzer= "transformer_attribution") # edit whatever other index want to look at here

    # axs[step, 1].imshow(dog_ro);
    # axs[step, 1].axis('off');
    # axs[step, 2].imshow(dog);
    # axs[step, 2].axis('off');
    # axs[step, 3].imshow(cat);
    # axs[step, 3].axis('off');

    plt.savefig('submission_model_A_norms'+'.png')
    plt.close()
    # exit()


# For Question 5 

# norm_sum = torch.zeros(6)
# bins = torch.zeros(3,7,7)
# indices = torch.arange(0,7,1)
# for step, (i,y) in enumerate(orig_dataloader):
#     output, token_norms = model(i.cuda(), return_token_norms=True) 
#     print(token_norms['layer_12'].shape)   
#     A, B = token_norms['layer_12'].shape 
#     layer_12 = token_norms['layer_12'].reshape(A,B).detach().cpu()
#     norm_sum += torch.sum(layer_12[:, :6], dim=0)
#     for im, layer in enumerate(token_norms):
#         print(layer)
#         norms = token_norms[f'{layer}']
#         # print(norms)
#         print(norms.shape)
#         print('Norms',norms.shape)
#         layer_x = norms.reshape(norms.shape[0], norms.shape[1])
#         o_ranks = torch.argsort(layer_x, dim = 1, descending = True)
#         print(o_ranks)
#         o_ranks = o_ranks.detach().cpu()

#         for ranks in o_ranks:
#             if layer == 'layer_1':
#                 l = 0
#             elif layer == 'layer_6':
#                 l = 1
#             elif layer == 'layer_12':
#                 l = 2
#             bins[l, ranks[:], indices[:]] += 1


# token_ranks = torch.argsort(norm_sum, descending=True) #ref for Tok A -> F
# fig , axs = plt.subplots(7, 3)
# label = np.array (['A', 'B', 'C', 'D', 'E', 'F'])
# for i in range(3):
#     layer_bin = bins[i]
#     layer_bin_norm = layer_bin/torch.sum(layer_bin, dim = 1)
#     for c in range(6):
#         axs[c,i].bar(np.arange(7), layer_bin_norm[token_ranks[c],:].numpy(), color = 'pink',edgecolor='black')
#         axs[c,i].tick_params(axis='both', which='major', labelsize=6)
#         axs[c, i].set_ylim([0, 1])

#     axs[6,i].bar(np.arange(7),layer_bin_norm[6,:].numpy(),color='pink',edgecolor='black')
#     axs[6,i].tick_params(axis='both', which='major', labelsize=6)
# plt.savefig('solution_norm_histograms.png')




