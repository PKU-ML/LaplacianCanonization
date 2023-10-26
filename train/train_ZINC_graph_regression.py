"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
from train.metrics import MAE


def handle_lap(model, batch_pos_enc, batch_graphs, device):
    if model.pe_init == 'lap_pe':  # random sign
        sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
    elif model.pe_init in ['map', 'map_ablation']:
        batch_pos_enc = batch_pos_enc.unsqueeze(-1)  # n x k -> k x n x 1
        batch_pos_enc = model.sign_inv_net(batch_graphs, batch_pos_enc)
        batch_pos_enc = batch_pos_enc.squeeze(-1)  # k x n x 1 -> n x k
    return batch_pos_enc


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device).float()
        except KeyError:
            batch_pos_enc = None
        if model.pe_init in ['lap_pe', 'map', 'map_ablation']:
            batch_pos_enc = handle_lap(model, batch_pos_enc, batch_graphs, device).float()
        batch_scores, __ = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        del __
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    return epoch_loss, epoch_train_mae, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    out_graphs_for_lapeig_viz = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device).float()
            except KeyError:
                batch_pos_enc = None
            if model.pe_init in ['map', 'map_ablation']:
                batch_pos_enc = handle_lap(model, batch_pos_enc, batch_graphs, device).float()
            batch_scores, batch_g = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            out_graphs_for_lapeig_viz += dgl.unbatch(batch_g)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
    return epoch_test_loss, epoch_test_mae, out_graphs_for_lapeig_viz
