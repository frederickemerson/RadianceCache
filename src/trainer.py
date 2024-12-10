import os

from src.model.sr_common import ELSR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from decimal import Decimal
import numpy as np
import utility

import torch
from tqdm import tqdm
from data import data_utils
import torch.nn as nn
from torch.utils.data import Dataset
from elsr import ELSR, train as train_elsr, validate as validate_elsr
from elsr.dataset import TrainDataset, ValDataset


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer, self).__init__()
        self.args = args
        self.scale = args.scale
        self.gt_size = args.gt_size
        self.batch_size = args.batch_size
        self.ckp = ckp
        self.model = my_model
        self.num_frames_samples = args.num_frames_samples
        self.train_loader = loader.loader_train
        self.valid_loader = loader.loader_valid
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        if args.use_elsr:
            self.ckp.write_log("Using ELSR as a preprocessing module...")
            self.elsr = ELSR(upscale_factor=args.scale).to(self.model.device)
            elsr_pretrained_path = args.elsr_path

            if not elsr_pretrained_path or not os.path.exists(elsr_pretrained_path):
                self.ckp.write_log("No pretrained elsr model found. Training elsr...")
                elsr_pretrained_path = self.train_elsr_model(args)  # Automatically train and fetch path

            self.elsr.load_state_dict(torch.load(elsr_pretrained_path))
            self.elsr.eval()
        else:
            self.elsr = None

        self.error_last = 1e8

    def train_elsr_model(self, args):
        self.ckp.write_log("No pretrained ELSR model found. Training ELSR...")

        # Create datasets and dataloaders
        train_dataset = TrainDataset(args.train_h5)
        val_dataset = ValDataset(args.val_h5)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

        # Initialize model, loss, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ELSR(upscale_factor=args.scale).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train and validate
        for epoch in range(1, args.elsr_epochs + 1):
            train_loss = train_elsr(model, train_loader, criterion, optimizer, device)
            val_psnr = validate_elsr(model, val_loader, device)
            self.ckp.write_log(
                f"ELSRAutoTrainer Epoch [{epoch}/{args.elsr_epochs}], Loss: {train_loss:.4f}, PSNR: {val_psnr:.4f}")

        # Save model
        elsr_path = os.path.join(args.ckpt_dir, "elsr_pretrained.pth")
        torch.save(model.state_dict(), elsr_path)
        self.ckp.write_log(f"ELSRAutoTrainer: ELSR model saved to {elsr_path}")
        return model

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        print(self.train_loader.n_samples)
        # TEMP
        for batch, (LR_lst, HR_lst, MV_up_lst, Mask_up_lst, _) in enumerate(self.train_loader):
            print(
                f"Batch {batch} - LR: {LR_lst[0].shape}, HR: {HR_lst[0].shape}, MV_up: {MV_up_lst[0].shape}, Mask_up: {Mask_up_lst[0].shape}")
            self.optimizer.zero_grad()

            b, c, h, w = HR_lst[0].size()
            zero_tensor = torch.zeros(b, c, h, w, dtype=torch.float32)
            lr0, zero_tensor, hr0 = self.prepare(LR_lst[0], zero_tensor, HR_lst[0])

            sr_pre, lstm_state = self.model((lr0, zero_tensor, None))
            lstm_state = utility.repackage_hidden(lstm_state)
            loss = self.loss(sr_pre, hr0, needTem=False)
            print(f"Initial loss for batch {batch}: {loss.item()}")

            for i in range(1, self.num_frames_samples):
                sr_pre = sr_pre.detach()
                sr_pre.requires_grad = False

                lr, hr, mv_up, mask_up = self.prepare(LR_lst[i], HR_lst[i], MV_up_lst[i], Mask_up_lst[i])

                timer_data.hold()
                timer_model.tic()

                sr_pre_warped = data_utils.warp(sr_pre, mv_up)
                sr_cur, lstm_state = self.model((lr, sr_pre_warped, lstm_state))
                lstm_state = utility.repackage_hidden(lstm_state)

                loss += self.loss(sr_cur, hr, sr_pre_warped, mask_up, needTem=True)
                sr_pre = sr_cur

                if self.args.use_elsr:
                    with torch.no_grad():
                        sr_elsr = self.elsr(lr0)
                    loss_elsr = self.loss(sr_elsr, hr0, needTem=False)
                    loss += 0.1 * loss_elsr

            loss.backward()
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    self.train_loader.n_samples,
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, 3)
        )
        self.model.eval()

        timer_test = utility.timer()
        run_model_time = 0
        flag = 0
        if self.args.save_results: self.ckp.begin_background()

        pre_sr = torch.zeros(1, 3, self.gt_size[0], self.gt_size[1],
                             dtype=torch.float32).cuda()
        lstm_state = None
        for index, (LR_lst, HR_lst, MV_up_lst, Mask_up_lst, filename) in tqdm(enumerate(self.valid_loader)):
            lr, hr, mv_up, mask_up = self.prepare(LR_lst[0], HR_lst[0], MV_up_lst[0], Mask_up_lst[0])
            if index == 0:
                pre_sr, lstm_state = self.model((lr, pre_sr, lstm_state))
                lstm_state = utility.repackage_hidden(lstm_state)
                continue
            t1 = time.time()
            sr_pre_warped = data_utils.warp(pre_sr, mv_up)
            cur_sr, lstm_state = self.model((lr, sr_pre_warped, lstm_state))
            if self.args.use_elsr:
                with torch.no_grad():
                    sr_elsr = self.elsr(lr)
                val_elsr_loss = self.loss(sr_elsr, hr, needTem=False)
                self.ckp.log[-1, 0] += val_elsr_loss.item()
            lstm_state = utility.repackage_hidden(lstm_state)
            t2 = time.time()
            run_model_time += (t2 - t1)
            if self.args.sr_content == "View":
                sr = utility.quantize_img(cur_sr)
                sr_last = utility.quantize_img(pre_sr)
                if flag < 2:
                    try:
                        print(f"Saving SR image to ./check/sr_{flag}.png")
                        data_utils.save2Exr(np.array(sr[0, :3, :, :].permute(1, 2, 0).detach().cpu()) * 255,
                                            f"./check/sr_{flag}.png")
                    except Exception as e:
                        print(f"Error saving SR image: {e}")
                    try:
                        print(f"Saving GT image to ./check/gt_{flag}.png")
                        data_utils.save2Exr(np.array(hr[0, :3, :, :].permute(1, 2, 0).detach().cpu()) * 255,
                                            f"./check/gt_{flag}.png")
                    except Exception as e:
                        print(f"Error saving GT image: {e}")
                    flag += 1
            else:
                sr = utility.quantize(cur_sr)
                sr_last = utility.quantize(pre_sr)
                if flag < 2:
                    try:
                        print(f"Saving SR image to ./check/sr_{flag}.exr")
                        data_utils.save2Exr(np.array(sr[0, :3, :, :].permute(1, 2, 0).detach().cpu()),
                                            f"./check/sr_{flag}.exr")
                    except Exception as e:
                        print(f"Error saving SR image: {e}")
                    try:
                        print(f"Saving GT image to ./check/gt_{flag}.exr")
                        data_utils.save2Exr(np.array(hr[0, :3, :, :].permute(1, 2, 0).detach().cpu()),
                                            f"./check/gt_{flag}.exr")
                    except Exception as e:
                        print(f"Error saving GT image: {e}")
                    flag += 1

            pre_sr = cur_sr
            save_list = [sr]
            assert sr is not torch.nan, "sr is nan!"
            val_ssim = 1.0 - utility.calc_ssim(sr, hr).cpu()
            warped_sr = data_utils.warp(sr_last, mv_up)
            val_tempory = utility.calc_tempory(warped_sr, sr, mask_up).cpu()

            self.ckp.log[-1, 0] += val_ssim
            self.ckp.log[-1, 1] += val_tempory
            self.ckp.log[-1, 2] += val_tempory + val_ssim

            if self.args.save_gt:
                save_list.extend([lr, hr])

            if self.args.save_results:
                self.ckp.save_results(self.valid_loader, filename[0], save_list, self.scale)

        self.ckp.log[-1] /= (len(self.valid_loader) - 1)
        best = self.ckp.log.min(0)
        self.ckp.write_log(
            '[{} x{}]\tSSIM: {:.6f}, Tempory: {:.6f}, Total :{:.6f} (Best: {:.6f} @epoch {})'.format(
                self.valid_loader.dataset.name,
                self.scale,
                self.ckp.log[-1][0],
                self.ckp.log[-1][1],
                self.ckp.log[-1][2],
                best[0][2],
                best[1][2] + 1
            )
        )

        self.ckp.write_log('Run model time {:.5f}s\n'.format(run_model_time))
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[0][2] is not torch.nan and best[1][2] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

    def train_elsr(self, args):
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        from model.sr_common import ELSR
        from PIL import Image

        # Create dataset class for LR-HR image pairs
        class LRDataset(Dataset):
            def __init__(self, lr_dir, hr_dir, transform=None):
                self.lr_dir = lr_dir
                self.hr_dir = hr_dir
                self.lr_images = sorted(os.listdir(lr_dir))
                self.hr_images = sorted(os.listdir(hr_dir))
                self.transform = transform

            def __len__(self):
                return len(self.lr_images)

            def __getitem__(self, idx):
                lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
                hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

                lr_image = transforms.ToTensor()(Image.open(lr_path).convert("RGB"))
                hr_image = transforms.ToTensor()(Image.open(hr_path).convert("RGB"))

                if self.transform:
                    lr_image = self.transform(lr_image)
                    hr_image = self.transform(hr_image)

                return lr_image, hr_image

        # Training function for elsr
        def run_elsr_training():
            self.ckp.write_log("Starting elsr training...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Prepare dataset and dataloader
            transform = transforms.Compose([transforms.Resize((64, 64))])  # Resize for uniformity
            dataset = LRDataset(args.lr_dir, args.hr_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            # Initialize elsr, loss, and optimizer
            elsr_model = ELSR(upscale_factor=args.scale).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(elsr_model.parameters(), lr=args.lr)

            # Training loop
            for epoch in range(args.elsr_epochs):
                total_loss = 0.0
                for lr, hr in dataloader:
                    lr, hr = lr.to(device), hr.to(device)
                    optimizer.zero_grad()
                    sr = elsr_model(lr)
                    loss = criterion(sr, hr)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.ckp.write_log(
                    f"ELSRAutoTrainer: Epoch [{epoch + 1}/{args.elsr_epochs}], Loss: {total_loss / len(dataloader):.4f}")

            # Save elsr model
            save_path = "elsr_pretrained_auto.pth"
            torch.save(elsr_model.state_dict(), save_path)
            self.ckp.write_log(f"ELSRAutoTrainer: elsr pretrained model saved to {save_path}")
            return save_path

        # Call the training function and return the path
        return run_elsr_training()
