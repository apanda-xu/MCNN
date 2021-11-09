import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json

from models.mcnn import MCNN


class trainer():
    def __init__(self, args, DataLoader):
        self.dataset = args.dataset
        self.DataLoader = DataLoader
        self.lr = float(args.lr)
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.device = args.device
        self.test_mode = args.test_mode
        self.resume = args.resume
        self.use_tensorboard = args.use_tensorboard

        self.train_path = args.train_path
        self.train_gt_path = args.train_gt_path
        self.val_path = args.val_path
        self.val_gt_path = args.val_gt_path
        self.test_path = args.test_path
        self.test_gt_path = args.test_gt_path

        self.saved_path = os.path.join(args.workspace, args.outdir, args.net, args.dataset)
        self.best_model_path = os.path.join(self.saved_path, "best_model.pth")
        self.record_path = os.path.join(self.saved_path, "record.json")
        self.test_result_path = os.path.join(self.saved_path, "test_result.txt")
        self.log_path = os.path.join(args.workspace, args.logdir, args.net, args.dataset)

        self.record = {}
        self.record['loss_history'] = []
        self.record['mae_history'] = []
        self.record['mse_history'] = []

        self.best_model = None
        self.best_mae = sys.maxsize
        self.best_mse = sys.maxsize

        self.net = self.load_net(args.net)

    # init net weight
    def weights_normal_init(self, model, dev=0.01):
        if isinstance(model, list):
            for m in model:
                self.weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)

    # load net
    def load_net(self, net):
        if net == "MCNN":
            net = MCNN().to(self.device)
            self.weights_normal_init(net, dev=0.01)
        else:
            print(">>> %s is not support!\n" % self.net)
            exit()
        # if training and resume, load latest saved model
        if not self.test_mode and self.resume:
            net = self.resume_from_lastest(net)
        # if test, load best model
        if self.test_mode:
            net.load_state_dict(torch.load(self.best_model_path))

        return net

    # resume from latest saved model
    def resume_from_lastest(self, net):
        # load record data
        with open(self.record_path, 'r') as f:
            self.record = json.load(f)
            f.close()
        self.start_epoch = self.record['last_epoch']+1
        self.best_mae = self.record['best_model_mae']
        self.best_mse = self.record['best_model_mse']
        # load latest saved model
        latest_saved_model_path = os.path.join(self.saved_path, str(self.record['last_epoch']) + ".pth")
        net.load_state_dict(torch.load(latest_saved_model_path))
        print(">>> resumed from ", latest_saved_model_path)
        return net

    # save infos while training
    def save_training(self):
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)
        # save best model
        torch.save(self.best_model, self.best_model_path)
        # save record data
        with open(self.record_path, 'w') as f:
            json.dump(self.record, f)
            f.close()
        # save latest model
        torch.save(self.net.state_dict(), os.path.join(
            self.saved_path, str(self.record['last_epoch'])+".pth"))

    # evaluate
    def evaluate(self, data_loader):
        self.net.eval()     # set eval
        mae = 0.0
        mse = 0.0
        with torch.no_grad():
            for blob in data_loader:
                im_data = torch.from_numpy(blob['data']).to(self.device)
                gt_data = torch.from_numpy(blob['gt_density']).to(self.device)
                density_map = self.net(im_data)
                gt_count = torch.sum(gt_data)
                et_count = torch.sum(density_map)
                mae += abs(gt_count-et_count)
                mse += pow(gt_count-et_count, 2)
            mae = mae / data_loader.get_num_samples()
            mse = torch.sqrt(mse / data_loader.get_num_samples())
        
        self.net.train()    # set train()
        return mae.item(), mse.item()

    # start training
    def train(self):
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_path)

        self.net.train()

        train_loader = self.DataLoader(self.train_path, self.train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
        val_loader = self.DataLoader(self.val_path, self.val_gt_path, shuffle=True, gt_downsample=True, pre_load=True)

        criterion = nn.MSELoss()
        optimiser = optim.Adam(self.net.parameters(), lr=self.lr)

        for epoch in range(self.start_epoch, self.max_epoch+1):
            epoch_loss = 0
            step = 0

            for blob in train_loader:
                im_data = torch.from_numpy(blob['data']).to(self.device)
                gt_data = torch.from_numpy(blob['gt_density']).to(self.device)

                density_map = self.net(im_data)
                loss = criterion(density_map, gt_data)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                step += 1

                # display evey 500 step
                if step % 500 == 0:
                    gt_count = torch.sum(gt_data)
                    et_count = torch.sum(density_map)
                    print("epoch: %d, step: %-4d, gt_count: %.2f, et_count: %.2f" % (epoch, step, gt_count, et_count))

            # evaluate on val data
            mae, mse = self.evaluate(val_loader)

            # update record data every epoch
            self.record['last_epoch'] = epoch
            self.record['lr'] = self.lr
            self.record['max_epochs'] = self.max_epoch
            self.record['loss_history'].append(epoch_loss)
            self.record['mae_history'].append(mae)
            self.record['mse_history'].append(mse)

            # update best model every epoch
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_mse = mse
                self.best_model = self.net.state_dict()
                self.record['best_model_id'] = epoch
                self.record['best_model_mae'] = mae
                self.record['best_model_mse'] = mse
            
            # print training infos
            print("EPOCH: %d, LOSS: %.2f, MAE: %.2f, MSE: %.2f" %(epoch, epoch_loss, mae, mse))
            print("BEST MODEL | ID: %d, MAE: %.2f, MSE: %.2f\n" % (
                self.record['best_model_id'], self.record['best_model_mae'], self.record['best_model_mse']))

            # save "record data", "latest model" and "best model" every 50 epochs
            if epoch % 100 == 0:
                self.save_training()

            # if use tensorboard, write logs
            if self.use_tensorboard:
                self.writer.add_scalar("Loss/mse_loss", epoch_loss, epoch)
                self.writer.add_scalar("Metrics/mae", mae, epoch)
                self.writer.add_scalar("Metrics/mse", mse, epoch)
                self.writer.add_scalar("Metrics/best_mae", self.best_mae, epoch)
                self.writer.add_scalar("Metrics/best_mse", self.best_mse, epoch)

    def test(self):
        test_loader = self.DataLoader(self.test_path, self.test_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
        self.test_mode = True
        mae, mse = self.evaluate(test_loader)
        res = "{:10s} MAE: {:.2f}, MSE: {:.2f}".format(self.dataset, mae, mse)
        print(">>> " + res)

        # save test result to txt
        with open(self.test_result_path, 'w') as f:
            f.write(res)
            f.close()