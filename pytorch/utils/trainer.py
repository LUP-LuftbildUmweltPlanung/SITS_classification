import torch
from pytorch.utils.classmetric import ClassMetric, RegressionMetric
from sklearn.metrics import roc_auc_score, auc
from pytorch.utils.printer import Printer

import sys
sys.path.append("../models")
sys.path.append("../models/transformer")
sys.path.append("../models/transformer/TransformerEncoder")
sys.path.append("models")
sys.path.append("..")

import os
import numpy as np
from pytorch.models.ClassificationModel import ClassificationModel
import torch.nn.functional as F
from pytorch.utils.scheduled_optimizer import ScheduledOptim
import copy
from tqdm import tqdm
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
                 norm_factor_response = None,
                 **kwargs):

        self.norm_factor_response = norm_factor_response
        self.response = response
        self.epochs = epochs
        self.batch_size = validdataloader.batch_size
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.dataset.dataset.nclasses
        self.sequencelength=traindataloader.dataset.dataset.dataset.sequencelength
        self.ndims = traindataloader.dataset.dataset.dataset.ndims
        self.store = store
        self.valid_every_n_epochs = valid_every_n_epochs
        self.logger = logger
        self.model = model
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.early_stopping_smooth_period = 5
        self.early_stopping_patience = 4
        self.not_improved_epochs=0
        self.trial = trial
        #self.early_stopping_metric="kappa"

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        #self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


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
        nclasses=self.nclasses,
        sequencelength=self.sequencelength,
        ndims=self.ndims,
        logged_data=self.logger.get_data())

    def fit(self):
        printer = Printer()

        while self.epoch < self.epochs:
            self.new_epoch() # increments self.epoch

            self.logger.set_mode("train")
            stats = self.train_epoch(self.epoch)
            #print(stats)
            self.logger.log(stats, self.epoch)
            printer.print(stats, self.epoch, prefix="\n"+"train: ")

            if self.epoch % self.valid_every_n_epochs == 0 or self.epoch==1:
                self.logger.set_mode("valid")
                stats = self.valid_epoch(self.validdataloader)
                #print(stats)
                self.logger.log(stats, self.epoch)
                printer.print(stats, self.epoch, prefix="\n"+"vali: ")
                print("")
                print("###"*10)

                # Add prune mechanism
                if self.trial:
                    if self.response == 'classification':
                        self.trial.report(stats['accuracy'], self.epoch)
                    else:
                        self.trial.report(stats['rmse'], self.epoch)

                    #if self.trial.should_prune():
                        #raise optuna.exceptions.TrialPruned()

            if self. epoch % self.checkpoint_every_n_epochs ==0:
                if not self.trial:
                    #print("Saving model to {}".format(self.get_model_name()))
                    self.snapshot(self.get_model_name())
                    #torch.save(self.model, self.get_model_name())
                    print("Saving log to {}".format(self.get_log_name()))
                    self.logger.get_data().to_csv(self.get_log_name())

            if self.epoch > self.early_stopping_smooth_period and self.check_for_early_stopping(smooth_period=self.early_stopping_smooth_period):
                if not self.trial:
                    print()
                    print(f"Model did not improve in the last {self.early_stopping_smooth_period} epochs. stopping training...")
                    print("Saving model to {}".format(self.get_model_name()))
                    self.snapshot(self.get_model_name())
                    #torch.save(self.model, self.get_model_name())
                    print("Saving log to {}".format(self.get_log_name()))
                    self.logger.get_data().to_csv(self.get_log_name())
                return self.logger


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

        for iteration, data in progress_bar:

            self.optimizer.zero_grad()
            inputs, targets, doy, thermal = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            if self.model.__class__.__name__ == "TransformerEncoder" and thermal is not None:
                logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2), doy, thermal)
            elif self.model.__class__.__name__ == "TransformerEncoder":
                logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2), doy)
            else:
                logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1,2))
            #print(inputs.transpose(1,2))
            #print(self.model(inputs.transpose(1,2))[0])
            if self.response == "classification":
                loss = F.nll_loss(logprobabilities, targets)
            else:
                loss = F.mse_loss(logprobabilities.squeeze(1),targets)

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
                accuracy_metrics = metric.update_confmat(targets.detach().cpu().numpy(),prediction.detach().cpu().numpy())
                stats["accuracy"] = accuracy_metrics["overall_accuracy"]
                stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()
                stats["mean_recall"] = accuracy_metrics["recall"].mean()
                stats["mean_precision"] = accuracy_metrics["precision"].mean()
                stats["mean_f1"] = accuracy_metrics["f1"].mean()
                stats["kappa"] = accuracy_metrics["kappa"]
            else:
                stats = metric.add(stats)
                # Calculate the MSE and R2 for this iteration
                targets = targets.detach().cpu().numpy()
                responses = logprobabilities.squeeze(1).detach().cpu().numpy()

                if self.norm_factor_response == "log10":
                    targets = np.power(10, targets) - 1
                    responses = np.power(10, responses) - 1
                elif self.norm_factor_response is not None and self.norm_factor_response != 0:
                    targets /= self.norm_factor_response
                    responses /= self.norm_factor_response


                rmse_r2_stats = metric.update_mat(targets,responses)

                # Add MSE and R2 to the `stats` dictionary
                stats["r2"] = rmse_r2_stats["r2"]
                stats["rmse"] = rmse_r2_stats["rmse"]


            # Updating the progress bar with the latest stats (e.g., loss)
            if self.response == "classification":
                progress_bar.set_postfix(loss=stats["loss"].item(), acc=stats["accuracy"], refresh=True)
            else:
                progress_bar.set_postfix(loss=stats["loss"].item(), acc=stats["rmse"], refresh=True)

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
            for iteration, data in enumerate(dataloader):

                inputs, targets, doy, thermal = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                if self.model.__class__.__name__ == "TransformerEncoder" and thermal is not None:
                    logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2), doy, thermal)
                elif self.model.__class__.__name__ == "TransformerEncoder":
                    logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2), doy)
                else:
                    logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2))

                if self.response == "classification":
                    loss = F.nll_loss(logprobabilities, targets)
                else:
                    loss = F.mse_loss(logprobabilities.squeeze(1),targets)

                stats = dict(
                    loss=loss,
                )


                if self.response == "classification":
                    prediction = self.model.predict(logprobabilities)
                    ## enter numpy world
                    prediction_np = prediction.detach().cpu().numpy()
                    label = targets.detach().cpu().numpy()
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
                    targets_vali = targets.detach().cpu().numpy()
                    responses_vali = logprobabilities.squeeze(1).detach().cpu().numpy()

                    if self.norm_factor_response == "log10":
                        targets_vali = np.power(10, targets_vali) - 1
                        responses_vali = np.power(10, responses_vali) - 1
                    elif self.norm_factor_response is not None and self.norm_factor_response != 0:
                        targets_vali /= self.norm_factor_response
                        responses_vali /= self.norm_factor_response

                    rmse_r2_stats = metric.update_mat(targets_vali,responses_vali)
                    # Add MSE and R2 to the `stats` dictionary
                    stats["r2"] = rmse_r2_stats["r2"]
                    stats["rmse"] = rmse_r2_stats["rmse"]


            stats["targets"] = targets.cpu().numpy()
            stats["inputs"] = inputs.cpu().numpy()

        return stats

