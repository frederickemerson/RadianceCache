import numpy as np
import torch
from elsr.preprocessing import psnr


def train(model, dataloader, loss_fn, optimizer, device, scheduler):
    model.train()
    train_loss = 0
    for i, data in enumerate(dataloader):
        lr, hr = data
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        np.save("plot_data/lr.npy", lr[0].cpu().numpy().transpose(1,2,0))
        np.save("plot_data/hr.npy", hr[0].cpu().numpy().transpose(1,2,0))
        np.save("plot_data/sr.npy", sr[0].detach().cpu().numpy().transpose(1,2,0))
        loss = loss_fn(sr, hr)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss

def validate(model, dataloader, device):
    model.eval()
    psnr_sum = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            lr, hr = data
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_sum += psnr(sr, hr)

    avg_psnr = psnr_sum / len(dataloader)
    return avg_psnr