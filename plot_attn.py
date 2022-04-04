"""Plot attention matrices by samples for each class
Compare noisy and non-noisy images
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

def plot_attn_clean_vs_noise(blocks, heads, n_samples, attn_clean, attn_noise, save_dir, n_classes, y):
    os.makedirs(save_dir, exist_ok=True)
    for c in range(n_classes):
        n_blocks = len(blocks)
        n_heads = len(heads)
        samples_idx = np.arange(len(y))[y==c][:n_samples]
        n_samples = len(samples_idx)
        print(n_blocks*n_heads, n_samples)
        fig, ax = plt.subplots(nrows=n_blocks*n_heads, ncols=n_samples, figsize=(n_samples*4, n_heads*n_blocks*4))
        ax[0, 0].set_title(f'class {c}')
        for i in range(len(samples_idx)):
            row = 0
            for b in blocks:
                for h in heads:
                    im = ax[row, i].imshow(attn_clean[b, samples_idx[i], h], vmin=0, vmax=1, cmap='coolwarm')
                    fig.colorbar(im, ax=ax[row, i])
                    ax[row, i].set_ylabel(f'block {b}, head {h}')
                    ax[row, i].set_title(f'class {c}, sample {i}')
                    row += 1
                    
        fig.savefig(os.path.join(save_dir, f'class{c}_clean_heatmap.pdf'))
        plt.close()
        
        fig, ax = plt.subplots(nrows=n_blocks*n_heads, ncols=n_samples, figsize=(n_samples*4, n_heads*n_blocks*4))
        ax[0, 0].set_title(f'class {c}')
        for i in range(len(samples_idx)):
            row = 0
            for b in blocks:
                for h in heads:
                    im = ax[row, i].imshow(attn_noise[b, samples_idx[i], h], vmin=0, vmax=1, cmap='coolwarm')
                    fig.colorbar(im, ax=ax[row, i])
                    ax[row, i].set_ylabel(f'block {b}, head {h}')
                    ax[row, i].set_title(f'class {c}, sample {i}')
                    row += 1
                    
        fig.savefig(os.path.join(save_dir, f'class{c}_noise_heatmap.pdf'))
        plt.close()
        
        fig, ax = plt.subplots(nrows=n_blocks*n_heads, ncols=n_samples, figsize=(n_samples*4, n_heads*n_blocks*4))
        ax[0, 0].set_title(f'class {c}')
        for i in range(len(samples_idx)):
            row = 0
            for b in blocks:
                for h in heads:
                    im = ax[row, i].imshow(torch.abs(attn_clean-attn_noise)[b, samples_idx[i], h], vmin=0, vmax=1, cmap='coolwarm')
                    fig.colorbar(im, ax=ax[row, i])
                    ax[row, i].set_ylabel(f'block {b}, head {h}')
                    ax[row, i].set_title(f'class {c}, sample {i}')
                    row += 1
                    
        fig.savefig(os.path.join(save_dir, f'class{c}_diff_heatmap.pdf'))
        plt.close()

def plot_attn_heatmap_per_sample(model_name, n_samples, attn_mat, save_dir, n_classes, y):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(n_samples):
        fig, ax = plt.subplots(nrows=MODEL_NUM_BLOCKS[model_name], ncols=MODEL_NUM_HEADS[model_name], figsize=(30, 50))
        for b in range(MODEL_NUM_BLOCKS[model_name]):
            for h in range(MODEL_NUM_HEADS[model_name]):
                im = ax[b, h].imshow(attn_mat[i, b, h], vmin=0, vmax=1, cmap='coolwarm')
                fig.colorbar(im, ax=ax[b, h])
                ax[b, h].set_title(f'block {b}, head {h}')
        fig.savefig(os.path.join(save_dir, f'class{y[i]}_sample{i}_heatmap.pdf'))
        plt.close()
        fig, ax = plt.subplots(nrows=MODEL_NUM_BLOCKS[model_name], ncols=MODEL_NUM_HEADS[model_name], figsize=(30, 50))
        for b in range(MODEL_NUM_BLOCKS[model_name]):
            for h in range(MODEL_NUM_HEADS[model_name]):
                ax[b, h].hist(attn_mat[i, b, h].reshape(-1).numpy())
                ax[b, h].set_xlim([0., 1.])
                ax[b, h].set_title(f'block {b}, head {h}')
        fig.savefig(os.path.join(save_dir, f'class{y[i]}_sample{i}_histogram.pdf'))
        plt.close()
        fig, ax = plt.subplots(nrows=MODEL_NUM_BLOCKS[model_name], ncols=MODEL_NUM_HEADS[model_name], figsize=(30, 50))
        for b in range(MODEL_NUM_BLOCKS[model_name]):
            for h in range(MODEL_NUM_HEADS[model_name]):
                ax[b, h].hist(attn_mat[i, b, h].reshape(-1).numpy(), bins=20, range=(0, 0.005))
                ax[b, h].set_xlim([0., 0.005])
                ax[b, h].set_title(f'block {b}, head {h}')
        fig.savefig(os.path.join(save_dir, f'class{y[i]}_sample{i}_histogram_thres0.005.pdf'))
        plt.close()

def main():
    ## hyperparameters
    model_name = "vit_small_patch16_224"
    img_size = 224
    noise_scale = 0.1
    n_samples = 10
    data = 'svhn'

    ## Data
    if data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_samples, shuffle=False)#, num_workers=8)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=n_samples, shuffle=False)#, num_workers=8)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
    elif data == 'svhn':
         ## Data
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        trainloader= torch.utils.data.DataLoader(trainset, batch_size=n_samples, shuffle=False)#, num_workers=8)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        testloader= torch.utils.data.DataLoader(testset, batch_size=n_samples, shuffle=False)#, num_workers=8)
        num_classes = 10
    # X, y = next(iter(trainloder))
    X, y = next(iter(testloader))

    ## Load network
    net = timm.create_model(model_name, pretrained=False)
    net.head = nn.Linear(net.head.in_features, 10)
    net = nn.DataParallel(net)
    checkpoint = torch.load(CKPT_PATHS[model_name], map_location='cpu')
    net.load_state_dict(checkpoint['model'])

    # create forward hooks
    layer_outputs = {}
    for block_i in range(MODEL_NUM_BLOCKS[model_name]):
        net.module.blocks[block_i].register_forward_hook(get_layer_outputs(f'block.{block_i}', layer_outputs))
        net.module.blocks[block_i].attn.register_forward_hook(get_layer_outputs(f'block.{block_i}.attn', layer_outputs))
        net.module.blocks[block_i].norm1.register_forward_hook(get_layer_outputs(f'block.{block_i}.norm1', layer_outputs))
        net.module.blocks[block_i].attn.qkv.register_forward_hook(get_layer_outputs(f'block.{block_i}.attn.qkv', layer_outputs))

    # forward pass samples
    print('true', y)
    X_clean = X
    X_noise = X + noise_scale * torch.randn_like(X)
    Z_clean = net(X_clean)
    attn_clean = get_attn_matries(model_name, net, layer_outputs).transpose(0, 1)
    print(attn_clean.shape)
    y_pred_clean = F.softmax(Z_clean, dim=1).argmax(1)
    print('pred_clean', y_pred_clean)
    print('acc clean', y_pred_clean.eq(y).sum() / len(y))
    # Z_noise = net(X_noise)
    # attn_noise = get_attn_matries(model_name, net, layer_outputs)
    # y_pred_noise = F.softmax(Z_noise, dim=1).argmax(1)
    # print('pred_noise', y_pred_noise)
    # print('acc noise', y_pred_noise.eq(y).sum() / len(y))

    # plot
    blocks = [0, 1, 2, 5, 6, 7, 9, 10, 11]
    # blocks = np.arange(model_num_blocks[model_name])
    heads = np.arange(MODEL_NUM_HEADS[model_name])
    # save_dir = f'./figures/{data}/{model_name}/attn_mat_per_noise_noise{noise_scale}'
    # plot_attn_clean_vs_noise(blocks, heads, n_samples, attn_clean, attn_noise, save_dir, num_classes, y)
    save_dir = f'./figures/{data}/{model_name}/attn_heatmap_per_sample/'
    plot_attn_heatmap_per_sample(model_name, n_samples, attn_clean, save_dir, num_classes, y)
if __name__ == '__main__':
    main()