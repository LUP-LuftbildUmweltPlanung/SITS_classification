import torch
from utils.classmetric import ClassMetric, RegressionMetric
from sklearn.metrics import roc_auc_score, auc
from utils.printer import Printer

import sys
sys.path.append("../models")
sys.path.append("../models/transformer")
sys.path.append("../models/transformer/TransformerEncoder")
sys.path.append("models")
sys.path.append("..")

import os
import numpy as np
from models.ClassificationModel import ClassificationModel
import torch.nn.functional as F
from utils.scheduled_optimizer import ScheduledOptim
import copy
from tqdm import tqdm


#import wandb
import optuna



class Trainer():

    def __init__(self,
                 trial,
                 model,
                 traindataloader,
                 validdataloader,
                 epochs=4,
                 learning_rate=0.1,
                 store="/tmp",
                 valid_every_n_epochs=1,
                 checkpoint_every_n_epochs=5,
                 optimizer=None,
                 logger=None,
                 response = None,
                 **kwargs):

        self.response = response
        self.epochs = epochs
        self.batch_size = validdataloader.batch_size
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.nclasses
        self.store = store
        self.valid_every_n_epochs = valid_every_n_epochs
        self.logger = logger
        self.model = model
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.early_stopping_smooth_period = 5
        self.early_stopping_patience = 5
        self.not_improved_epochs=0
        self.trial = trial
        #self.early_stopping_metric="kappa"
        ####self.test_file = 'test.txt' # REMOVE!

        #wandb.init(project="test_sits_main",config=kwargs) ##############################
        #wandb.watch(model, log_freq=100) ##################################

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        #self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        self.classweights = torch.FloatTensor(traindataloader.dataset.classweights)

        if torch.cuda.is_available():
            self.classweights = self.classweights.cuda()


        # only save checkpoint if not previously resumed from it
        self.resumed_run = False

        self.epoch = 0


    def resume(self, filename):
        snapshot = self.model.load(filename)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.epoch = snapshot["epoch"]
        print("resuming optimizer state")
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.logger.resume(snapshot["logged_data"])

    def snapshot(self, filename):
        self.model.save(
        filename,
        optimizer_state_dict=self.optimizer.state_dict(),
        epoch=self.epoch,
        logged_data=self.logger.get_data())

    def fit(self):
        printer = Printer()

        while self.epoch < self.epochs:
            self.new_epoch() # increments self.epoch

            self.logger.set_mode("train")
            stats = self.train_epoch(self.epoch)
            #print(stats)
            self.logger.log(stats, self.epoch)
            printer.print(stats, self.epoch, prefix="\n"+self.traindataloader.dataset.partition+": ")
            #wandb.log({"loss": stats['avg loss'],'epoch':self.epoch}) ##################################

            if self.epoch % self.valid_every_n_epochs == 0 or self.epoch==1:
                self.logger.set_mode("valid")
                stats = self.valid_epoch(self.validdataloader)
                #print(stats)
                self.logger.log(stats, self.epoch)
                printer.print(stats, self.epoch, prefix="\n"+self.validdataloader.dataset.partition+": ")
                #wandb.log({"val_loss": stats['avg loss'],'epoch':self.epoch}) ##################################
                print("")
                print("###"*10)

            if self. epoch % self.checkpoint_every_n_epochs ==0:
                print("Saving model to {}".format(self.get_model_name()))
                self.snapshot(self.get_model_name())
                print("Saving log to {}".format(self.get_log_name()))
                self.logger.get_data().to_csv(self.get_log_name())

            if self.epoch > self.early_stopping_smooth_period and self.check_for_early_stopping(smooth_period=self.early_stopping_smooth_period):
                print()
                print(f"Model did not improve in the last {self.early_stopping_smooth_period} epochs. stopping training...")
                print("Saving model to {}".format(self.get_model_name()))
                self.snapshot(self.get_model_name())
                print("Saving log to {}".format(self.get_log_name()))
                self.logger.get_data().to_csv(self.get_log_name())
                #wandb.finish() ##################################
                return self.logger

        #wandb.finish() ##################################
        #print(stats)
            # Add prune mechanism
            if self.response=='classification':
                self.trial.report(stats['accuracy'], self.epoch)
            else:
                self.trial.report(stats['rmse'], self.epoch)

            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return self.logger

    def check_for_early_stopping(self,smooth_period):
        log = self.logger.get_data()
        log = log.loc[log["mode"] == "valid"]

        early_stopping_condition = log["loss"].diff()[-smooth_period:].mean() > 0

        if early_stopping_condition:
            self.not_improved_epochs += 1
            print()
            print(f"model did not improve: {self.not_improved_epochs} of {self.early_stopping_patience} until early stopping...")
            return self.not_improved_epochs >= self.early_stopping_patience
        else:
            self.not_improved_epochs = 0
            return False


    def new_epoch(self):
        self.epoch += 1

    def get_model_name(self):
        return os.path.join(self.store, f"model_e{self.epoch}.pth")

    def get_log_name(self):
        return os.path.join(self.store, "log.csv")



    def train_epoch(self, epoch):
        # sets the model to train mode: dropout is applied
        self.model.train()

        if self.response == "classification":
            # builds a confusion matrix
            metric = ClassMetric(num_classes=self.nclasses)
        else:
            # Create an instance of RegressionMetric
            metric = RegressionMetric()

        # Wrap your data loader with tqdm to get a progress bar
        progress_bar = tqdm(enumerate(self.traindataloader),
                            total=len(self.traindataloader),
                            desc=f"Epoch {epoch}",
                            leave=True)

        cumu_loss = 0 ##################################

        for iteration, data in progress_bar:

            self.optimizer.zero_grad()

            inputs, targets, _ = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1,2))

            if self.response == "classification":
                loss = F.nll_loss(logprobabilities, targets[:, 0])
            else:
                loss = F.mse_loss(logprobabilities.squeeze(1),targets[:, 0])

            cumu_loss += loss.item() ##################################
            #print("loss="+str(loss.item())+', cumu_loss='+str(cumu_loss))

            loss.backward()

            stats = dict(
                loss=loss,
            )


            if isinstance(self.optimizer,ScheduledOptim):
                self.optimizer.step_and_update_lr()
            else:
                self.optimizer.step()


            #print("Shapes:", targets.shape, prediction.shape)
            if self.response == "classification":
                prediction = self.model.predict(logprobabilities)
                stats = metric.add(stats)
                accuracy_metrics = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(),prediction.detach().cpu().numpy())
                stats["accuracy"] = accuracy_metrics["overall_accuracy"]
                stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()
                stats["mean_recall"] = accuracy_metrics["recall"].mean()
                stats["mean_precision"] = accuracy_metrics["precision"].mean()
                stats["mean_f1"] = accuracy_metrics["f1"].mean()
                stats["kappa"] = accuracy_metrics["kappa"]
            else:
                stats = metric.add(stats)
                # Calculate the MSE and R2 for this iteration
                rmse_r2_stats = metric.update_mat(targets[:, 0].detach().cpu().numpy(),logprobabilities.squeeze(1).detach().cpu().numpy())
                # Add MSE and R2 to the `stats` dictionary
                stats["r2"] = rmse_r2_stats["r2"]
                stats["rmse"] = rmse_r2_stats["rmse"]


            # Updating the progress bar with the latest stats (e.g., loss)
            if self.response == "classification":
                progress_bar.set_postfix(loss=stats["loss"].item(), acc=stats["accuracy"], refresh=True)
            else:
                progress_bar.set_postfix(loss=stats["loss"].item(), acc=stats["rmse"], refresh=True)

            #wandb.log({"batch loss": loss}) ##################################

        stats['avg loss'] = cumu_loss/len(self.traindataloader) ##################################

        return stats

    def valid_epoch(self, dataloader, epoch=None):
        # sets the model to eval mode: no dropout is applied
        self.model.eval()
        if self.response == "classification":
            metric = ClassMetric(num_classes=self.nclasses)
        else:
            # Create an instance of RegressionMetric
            metric = RegressionMetric()


        with torch.no_grad():

            cumu_loss = 0 ##################################

            for iteration, data in enumerate(dataloader):

                inputs, targets, _ = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2))

                if self.response == "classification":
                    loss = F.nll_loss(logprobabilities, targets[:, 0])
                else:
                    loss = F.mse_loss(logprobabilities.squeeze(1), targets[:, 0])

                cumu_loss += loss.item() ##################################
                #print("loss="+str(loss.item())+', cumu_loss='+str(cumu_loss))

                stats = dict(
                    loss=loss,
                )


                if self.response == "classification":
                    prediction = self.model.predict(logprobabilities)
                    ## enter numpy world
                    prediction_np = prediction.detach().cpu().numpy()
                    label = targets.mode(1)[0].detach().cpu().numpy()
                    stats = metric.add(stats)
                    accuracy_metrics = metric.update_confmat(label, prediction_np)
                    stats["accuracy"] = accuracy_metrics["overall_accuracy"]
                    stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()
                    stats["mean_recall"] = accuracy_metrics["recall"].mean()
                    stats["mean_precision"] = accuracy_metrics["precision"].mean()
                    stats["mean_f1"] = accuracy_metrics["f1"].mean()
                    stats["kappa"] = accuracy_metrics["kappa"]
                else:
                    stats = metric.add(stats)
                    # Calculate the MSE and R2 for this iteration
                    rmse_r2_stats = metric.update_mat(targets[:, 0].detach().cpu().numpy(),logprobabilities.squeeze(1).detach().cpu().numpy())
                    # Add MSE and R2 to the `stats` dictionary
                    stats["r2"] = rmse_r2_stats["r2"]
                    stats["rmse"] = rmse_r2_stats["rmse"]

                #wandb.log({"batch val_loss": loss}) ##################################

            stats["targets"] = targets.cpu().numpy()
            stats["inputs"] = inputs.cpu().numpy()

            stats['avg loss'] = cumu_loss/len(dataloader) ##################################

        return stats

