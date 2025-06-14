import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import wandb

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        # wandb 초기화
        wandb.init(project='RFDN-SuperResolution', name=args.save)
        wandb.config.update(args)

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        # tqdm 적용
        pbar = tqdm(self.loader_train, desc=f'Epoch {epoch}/{self.args.epochs}', ncols=80)

        for batch, (lr, hr, _,) in enumerate(pbar):
            lr, hr = self.prepare(lr, hr)

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()

            # tqdm과 wandb에 loss 로깅
            pbar.set_postfix(Loss=f'{loss.item():.4f}')
            wandb.log({'Train/Loss': loss.item(), 'Epoch': epoch, 'Step': batch})

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()

        avg_psnr = 0
        num_samples = 0

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)

                pbar = tqdm(d, desc=f'Test Dataset: {d.dataset.name}', ncols=80)

                for lr, hr, filename in pbar:
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    avg_psnr += psnr
                    num_samples += 1

                    if self.args.save_results:
                        save_list = [sr]
                        if self.args.save_gt:
                            save_list.extend([lr, hr])
                        self.ckp.save_results(d, filename[0], save_list, scale)

                avg_psnr /= num_samples

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f}'.format(
                        d.dataset.name, scale, avg_psnr
                    )
                )

                wandb.log({'Test/PSNR': avg_psnr, 'Epoch': epoch})

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=True)

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
