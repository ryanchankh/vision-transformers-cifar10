"""Extract attention maps from last block
compare the mean response for in-distribution samples and out-distribution samples
"""


import os
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


CKPT_PATHS = {
    'vit_large_patch16_224': './checkpoint/vit_timm-4-ckpt-27ltmqh7.t7',
    'vit_base_patch16_224': './checkpoint/vit_timm-4-ckpt-119k9xmd.t7',
    'vit_small_patch16_224': './checkpoint/vit_timm-4-ckpt-2463dm3b.t7'
}

MODEL_NUM_BLOCKS = {
    'vit_large_patch16_224': 24,
    'vit_base_patch16_224': 12,
    'vit_small_patch16_224': 12
}

MODEL_NUM_HEADS = {
    'vit_large_patch16_224': 24,
    'vit_base_patch16_224': 12,
    'vit_small_patch16_224': 6,
}


def get_attn_matries(model_name, net, layer_outputs):
    attns = []
    for block_i in range(MODEL_NUM_BLOCKS[model_name]):
        x = layer_outputs[f'block.{block_i}.norm1']
        B, N, C = x.shape
        num_heads = net.module.blocks[block_i].attn.num_heads
        scale = net.module.blocks[block_i].attn.scale
        qkv_x = layer_outputs[f'block.{block_i}.attn.qkv']
        qkv = qkv_x.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attns.append(attn)
    return torch.stack(attns)

def get_layer_outputs(name, layer_outputs):
    def hook(model, input, output):
        layer_outputs[name] = output.detach()
    return hook

def entropy(p, dim=-1, eps=1e-5):
    return - ((p + eps) * torch.log((p + eps))).sum(dim)

def summarize_attn(method, attn_mat):
    if method == 'mean_sum':
        return attn_mat.mean(dim=-2).sum(dim=-1)
    elif method == 'sum_sum':
        return attn_mat.sum(dim=-2).sum(dim=-1)
    elif method == 'entropy_sum':
        return entropy(attn_mat, dim=-1).sum(dim=-1)
    elif method == 'entropy_mean':
        return entropy(attn_mat, dim=-1).mean(dim=-1)
    else:
        raise ValueError(f'method not found: {method}')

def plot_distributions(dist1, dist2, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n_samples, n_heads = dist1.shape
    fig, ax = plt.subplots(ncols=n_heads, figsize=(16, 4))
    for h in range(n_heads):
        ax[h].hist(dist1[:, h], bins=20, color='blue', alpha=0.5, label='cifar10')
        ax[h].hist(dist2[:, h], bins=20, color='red', alpha=0.5, label='svhn')
        ax[h].set_title(f'head: {h}')
    ax[-1].legend()
    fig.savefig(os.path.join(save_dir, 'cifar10_svhn.pdf'))

def main():
    ## hyperparameters
    model_name = "vit_small_patch16_224"
    img_size = 224
    batch_size = 128
    n_samples = 5000
    block = 11
    method = 'entropy_sum'

    ## device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)

    ## Data
    transform1 = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset1 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform1)
    testloader1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False)#, num_workers=8)
    num_classes = 10
    transform2 = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])
    testset2 = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform2)
    testloader2 = torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=False)#, num_workers=8)
    num_classes = 10

    ## Load network
    net = timm.create_model(model_name, pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
    net = nn.DataParallel(net)
    checkpoint = torch.load(CKPT_PATHS[model_name], map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    net = net.to(device)

    ## Create forward hooks
    layer_outputs = {}
    for block_i in range(MODEL_NUM_BLOCKS[model_name]):
        net.module.blocks[block_i].register_forward_hook(get_layer_outputs(f'block.{block_i}', layer_outputs))
        net.module.blocks[block_i].attn.register_forward_hook(get_layer_outputs(f'block.{block_i}.attn', layer_outputs))
        net.module.blocks[block_i].norm1.register_forward_hook(get_layer_outputs(f'block.{block_i}.norm1', layer_outputs))
        net.module.blocks[block_i].attn.qkv.register_forward_hook(get_layer_outputs(f'block.{block_i}.attn.qkv', layer_outputs))

    net.eval()
    attn_mats1 = []
    for batch_i, (X_test, y_test) in enumerate(testloader1):
        X_test = X_test.to(device)

        with torch.no_grad():
            Z_test = net(X_test)
            attn = get_attn_matries(model_name, net, layer_outputs).transpose(0, 1)
            print('attn.shape', attn.shape)
        attn_mats1.append(attn.cpu())
        if batch_i * batch_size  > n_samples:
            break
    attn_mats1 = torch.cat(attn_mats1, dim=0)
    print('attn mat1.shape', attn_mats1.shape)
    attn_mats2 = []
    for batch_i, (X_test, y_test) in enumerate(testloader2):
        X_test = X_test.to(device)

        with torch.no_grad():
            Z_test = net(X_test)
            attn = get_attn_matries(model_name, net, layer_outputs).transpose(0, 1)
            print('attn.shape', attn.shape)
        attn_mats2.append(attn.cpu())
        if batch_i * batch_size  > n_samples:
            break
    attn_mats2 = torch.cat(attn_mats2, dim=0)
    print('attn mat2.shape', attn_mats2.shape)


    dist_attn_mats1 = summarize_attn(method, attn_mats1[:, block, :, :, :]).numpy()
    dist_attn_mats2 = summarize_attn(method, attn_mats2[:, block, :, :, :]).numpy()
    # print(dist_attn_mats1)
    # print(dist_attn_mats2)
    save_dir = f'./figures/{model_name}/distribution_hist/'
    plot_distributions(dist_attn_mats1, dist_attn_mats2, save_dir)

if __name__ == '__main__':
    main()


