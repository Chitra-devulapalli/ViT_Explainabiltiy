import argparse
import os
import time
import torch
import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP


# TODO: CHANGE THIS CODE TO SINGLE-GPU 

def setup(rank, world_size, use_gloo=False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'gloo' if use_gloo else 'nccl'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def prepare(batch_size=64, pin_memory=False, num_workers=0):
    # traindir = '/nfs/turbo/coe-stellayu/shared_data/CUB_200_2011'
    traindir = os.path.join('/scratch/eecs542f24_class_root/eecs542f24_class/shared_data/imagenet-100', 'train')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    # augmentations = [
    #     transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ]

    # augmentations = [
    #     transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    #     ], p=0.5),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ]

    augmentations = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]

    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(augmentations))
    
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=True)
    # drop_last = False means pad, adding 0 as input, but else get cut off data to make divisble by world_size
    #REMOVED SAMPLER ARGUMENT FROM DATALOADER TO MAKE IT SINGLE GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=True)
    # print('DATASETTOCLASS', dataset.class_to_idx)
    # exit()
    return dataset, dataloader


def valprepare(batch_size=64, pin_memory=False, num_workers=0):
    # traindir = '/nfs/turbo/coe-stellayu/shared_data/CUB_200_2011'
    traindir = os.path.join('/scratch/eecs542f24_class_root/eecs542f24_class/shared_data/imagenet-100', 'val') #-100
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentations = [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]

    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(augmentations))
    
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=0, drop_last=True)
    # drop_last = False means pad, adding 0 as input, but else get cut off data to make divisble by world_size
    #REMOVED SAMPLER ARGUMENT FROM DATALOADER TO MAKE IT SINGLE GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=True)
    # print('DATASETTOCLASS', dataset.class_to_idx)
    # exit()
    return dataset, dataloader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad(): ## we don't do this because want to register grads
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

def validate(model, dataloader, device): 
    model.eval()
    # with torch.no_grad(): ## we don't do this because want to register grads
    acc1_total = 0
    acc5_total = 0
    total_items = 0
    for step, (x, y) in tqdm(enumerate(dataloader)):
        outputs = model(x.to(device))
        acc1, acc5 = accuracy(outputs, y.to(device), topk=(1, 5))
        
        acc1_total += acc1*len(y)
        acc5_total += acc5*len(y)
        total_items += len(y)

    total_acc1_tensor = torch.tensor(acc1_total, dtype=torch.float32, device=device)
    total_acc5_tensor = torch.tensor(acc5_total, dtype=torch.float32, device=device)
    total_items_tensor = torch.tensor(total_items, dtype=torch.float32, device=device)

    # dist.all_reduce(total_acc1_tensor, op=dist.ReduceOp.SUM)
    # dist.all_reduce(total_acc5_tensor, op=dist.ReduceOp.SUM)
    # dist.all_reduce(total_items_tensor, op=dist.ReduceOp.SUM)

    # if rank == 0:
    print(f"VALIDATION | Top 1 Acc: {total_acc1_tensor/total_items_tensor}, Top 5 Acc: {total_acc5_tensor/total_items_tensor}")
    

def demo_ddp(rank, world_size, use_gloo=False, epochs=8, scale=1, lr=0.01, warmup=5): #12 epoch
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size, use_gloo)
    print('Rank {} successful setup'.format(rank))

    torch.cuda.set_device(rank)
    model = vit_LRP(pretrained=True, num_classes=100).to(rank)

    ddp_model = DDP(model, device_ids=[rank])
    print('Rank {} successful ddp_model'.format(rank))
    
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # state_dict = torch.load('TEMP_NAME.pth', map_location=map_location) 
    
   
    ## EXAMPLE FORMAT IF MISMATCH IN WEIGHTS FILE AND MODEL

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if k == 'module.register_tokens' or k == 'module.pos_embedding': 
    #         print("DON'T initialize reg token and pos embeddings")
    #     else: 
    #         new_state_dict[k] = v

    # ddp_model.load_state_dict(new_state_dict, strict=False)


    dataset, dataloader = prepare(rank, world_size)
    valset, valloader = valprepare(rank, world_size)
    print('Rank {} past dataloaders'.format(rank))
    print('Rank {} dataloader length {}'.format(rank, len(dataloader)))
    
    # validate(ddp_model, valloader, rank)
    # exit()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)
   

    for epoch in range(epochs):
        if epoch<warmup:
            for g in optimizer.param_groups:
                g['lr'] = (epoch+1)*lr/(warmup+1)
                if rank == 0:
                    print((epoch+1)*lr/(warmup+1))
        else: 
            for g in optimizer.param_groups:
                g['lr'] = lr
        
        ddp_model.train()
        if rank == 0:
            print(f'Epoch: {epoch}')
            start_time = time.time()
        dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        total_items = 0
        for step, (x, y) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            outputs = ddp_model(x.to(rank))
            loss = loss_fn(outputs.to(rank), y.to(rank))
            
            loss_scaled = loss*scale
            loss_scaled.backward()
            optimizer.step()

            total_loss += loss.item()*len(y)
            total_items += len(y)

        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=rank)
        total_items_tensor = torch.tensor(total_items, dtype=torch.float32, device=rank)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_items_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            torch.save(ddp_model.state_dict(), 'TEMP_NAME.pth') ## EDIT HERE TO SAVE DIFFERENT THINGS
            print(f"Avg Epoch Loss: {total_loss_tensor/total_items_tensor}")
            print(f'Epoch Time: {time.time()-start_time}')
            
        if epoch%3 == 0: #validate
            validate(ddp_model, valloader, rank)
            
            
    cleanup()


def single_gpu(epochs=15, scale=1, lr=0.01, warmup=5): #12 epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vit_LRP(pretrained=True, num_classes=100).to(device)

    #loading weights from epoch 3
    
    # checkpoint = torch.load('/scratch/eecs542f24_class_root/eecs542f24_class/chitrakd/homework1/Model_B_3_epochs.pth', map_location=device)

    # model.load_state_dict(checkpoint, strict=True)

    dataset, dataloader = prepare()
    valset, valloader = valprepare()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
   

    for epoch in range(epochs):
        if epoch<warmup:
            for g in optimizer.param_groups:
                g['lr'] = (epoch+1)*lr/(warmup+1)
        else: 
            for g in optimizer.param_groups:
                g['lr'] = lr
        
        model.train()
        print(f"Epoch: {epoch}")
        start_time = time.time()

        total_loss = 0
        total_items = 0
        for step, (x, y) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            
            #UNCOMMENT FOR NORMS
            # output, token_norms = model(x.to(device), return_token_norms=True)
            outputs = model(x.to(device))
            loss = loss_fn(outputs.to(device), y.to(device))
            
            loss_scaled = loss*scale
            loss_scaled.backward()
            optimizer.step()

            total_loss += loss.item()*len(y)
            total_items += len(y)

        # total_loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=rank)
        # total_items_tensor = torch.tensor(total_items, dtype=torch.float32, device=rank)
        
        # dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(total_items_tensor, op=dist.ReduceOp.SUM)
        
        # if rank == 0:
        #     torch.save(ddp_model.state_dict(), 'TEMP_NAME.pth') ## EDIT HERE TO SAVE DIFFERENT THINGS
        #     print(f"Avg Epoch Loss: {total_loss_tensor/total_items_tensor}")
        #     print(f'Epoch Time: {time.time()-start_time}')

        avg_loss = total_loss / total_items
        print(f"Avg Epoch Loss: {avg_loss}")
        print(f"Epoch Time: {time.time() - start_time}")
            
        if epoch%3 == 0: #validate        print(f"Avg Epoch Loss: {avg_loss}")
            validate(model, valloader, device)

        torch.save(model.state_dict(), f'Model_B_final.pth')


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--use-gloo", action="store_true")
    # args = parser.parse_args()
    # print(args)
    # world_size = torch.cuda.device_count()
    # start_time = time.time()
    print("starting single gpu training")
    single_gpu()
    # mp.spawn(demo_ddp, nprocs=world_size, args=(world_size, args.use_gloo))
    # print('All ranks finished in {:.2f} seconds'.format(time.time() - start_time))


# USED WITH 2 GPU FOR FINETUNING

if __name__ == "__main__":
    
    print(torch.version.cuda)
    print(torch.__version__)
    torch.cuda.empty_cache()
    print("emptied cache")
    main()