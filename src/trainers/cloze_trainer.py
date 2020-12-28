import math
import os
import torch
from src.modules.optimizers import *
from src.modules.embeddings import *
from src.modules.schedulers import *
from src.modules.tokenizers import *
from src.modules.metrics import *
from src.modules.losses import *
from src.utils.misc import *
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.configuration import Config

from torch.utils.data import DataLoader
from tqdm import tqdm


@configmapper.map("trainers", "cloze")
class ClozeTrainer:
    def __init__(self, config):
        self._config = config
        self.metrics = [
            configmapper.get_object("metrics", metric)
            for metric in self._config.main_config.metrics
        ]
        self.train_config = self._config.train
        self.val_config = self._config.val
        self.log_label = self.train_config.log.log_label
        if self.train_config.log_and_val_interval is not None:
            self.val_log_together = True
        print("Logging with label: ", self.log_label)

    def train(self, model, train_dataset, val_dataset=None):
        device = torch.device(self._config.main_config.device.name)
        model.to(device)
        optim_params = self.train_config.optimizer.params
        if optim_params:
            optimizer = configmapper.get_object(
                "optimizers", self.train_config.optimizer.type
            )(model.parameters(), **optim_params.as_dict())
        else:
            optimizer = configmapper.get_object(
                "optimizers", self.train_config.optimizer.type
            )(model.parameters())

        if self.train_config.scheduler is not None:
            scheduler_params = self.train_config.scheduler.params
            if scheduler_params:
                scheduler = configmapper.get_object(
                    "schedulers", self.train_config.scheduler.type
                )(optimizer, **scheduler_params.as_dict())
            else:
                scheduler = configmapper.get_object(
                    "schedulers", self.train_config.scheduler.type
                )(optimizer)

        criterion_params = self.train_config.criterion.params
        if criterion_params:
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )(**criterion_params.as_dict())
        else:
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )()

        train_loader = DataLoader(
            train_dataset,
            **self.train_config.loader_params.as_dict(),
            collate_fn=train_dataset.custom_collate_fn,
        )
        # train_logger = Logger(**self.train_config.log.logger_params.as_dict())

        max_epochs = self.train_config.max_epochs
        batch_size = self.train_config.loader_params.batch_size

        if self.val_log_together:
            val_interval = self.train_config.log_and_val_interval
            log_interval = val_interval
        else:
            val_interval = self.train_config.val_interval
            log_interval = self.train_config.log.log_interval

        train_logger = Logger(**self.train_config.log.logger_params.as_dict())
        train_log_values = self.train_config.log.values.as_dict()

        best_score = (
            -math.inf if self.train_config.save_on.desired == "max" else math.inf
        )
        save_on_score = self.train_config.save_on.score
        best_step = -1
        best_model = None

        print("\nTraining\n")
        # print(max_steps)

        global_step = 0
        for epoch in range(1, max_epochs + 1):
            print(
                "Epoch: {}/{}, Global Step: {}".format(epoch, max_epochs, global_step)
            )
            train_loss = 0
            val_loss = 0

            all_labels = torch.FloatTensor().to(device)
            all_outputs = torch.Tensor().to(device)

            pbar = tqdm(total=math.ceil(len(train_dataset) / batch_size))
            pbar.set_description("Epoch " + str(epoch))

            val_counter = 0

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                batch = [torch.tensor(value, device=device) for value in batch]
                # print(batch[0].shape,batch)
                *inputs, labels = batch
                # print(inputs[0],inputs[1])
                # labels = labels.float()
                outputs = model(inputs)
                # print(outputs,labels)
                loss = criterion(outputs, labels)
                loss.backward()

                all_labels = torch.cat((all_labels, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)

                train_loss += loss.item()
                optimizer.step()

                if self.train_config.scheduler is not None:
                    scheduler.step(epoch + i / len(train_loader))

                # print(train_loss)
                # print(step+1)

                pbar.set_postfix_str(f"Train Loss: {train_loss /(step+1)}")
                pbar.update(1)

                global_step += 1

                # Need to check if we want global_step or local_step

                if val_dataset is not None and (global_step) % val_interval == 0:
                    print("\nEvaluating\n")
                    val_scores = self.val(
                        model,
                        val_dataset,
                        criterion,
                        device,
                        global_step,
                        train_logger,
                        train_log_values,
                    )
                    save_flag = 0
                    if self.train_config.save_on is not None:
                        train_loss_name = self.train_config.criterion.type
                        training_loss = train_loss / global_step

                        metric_list = [
                            metric(all_outputs.detach().cpu(), all_labels.cpu())
                            for metric in self.metrics
                        ]
                        metric_name_list = [
                            metric for metric in self._config.main_config.metrics
                        ]

                        train_scores = dict(
                            zip(
                                [train_loss_name,] + metric_name_list,
                                [training_loss,] + metric_list,
                            )
                        )

                        if self.train_config.save_on.desired == "min":
                            if val_scores[save_on_score] < best_score:
                                save_flag = 1
                                best_score = val_scores[save_on_score]
                                best_step = global_step
                        else:
                            if val_scores[save_on_score] > best_score:
                                save_flag = 1
                                best_score = val_scores[save_on_score]
                                best_step = global_step
                        if save_flag:
                            torch.save(
                                {
                                    "model_state_dict": model,
                                    "best_step": best_step,
                                    "best_score": best_score,
                                    "save_on_score": save_on_score,
                                },
                                self.train_config.save_on.best_path.format(
                                    self.log_label
                                ),
                            )

                            hparam_list = []
                            hparam_name_list = []
                            if self.train_config.log.values.hparams is not None:
                                for hparam in self.train_config.log.values.hparams:
                                    hparam_list.append(
                                        get_item_in_config(self._config, hparam["path"])
                                    )
                                    hparam_name_list.append(hparam["name"])

                                val_keys, val_values = zip(*val_scores.items())
                                train_keys, train_values = zip(*train_scores.items())
                                val_keys = list(val_keys)
                                train_keys = list(train_keys)
                                val_values = list(val_values)
                                train_values = list(train_values)
                                for i, key in enumerate(val_keys):
                                    val_keys[i] = (
                                        f"hparams/{self.log_label}/best_val_val_"
                                        + val_keys[i]
                                    )
                                for i, key in enumerate(train_keys):
                                    train_keys[i] = (
                                        f"hparams/{self.log_label}/best_val_train_"
                                        + train_keys[i]
                                    )
                                train_logger.save_hyperparams(
                                    hparam_list,
                                    hparam_name_list,
                                    train_values + val_values,
                                    train_keys + val_keys,
                                )

                if (global_step - 1) % log_interval == 0:
                    print("\nLogging\n")

                    train_loss_name = self.train_config.criterion.type
                    metric_list = [
                        metric(outputs.detach().cpu(), labels.cpu())
                        for metric in self.metrics
                    ]
                    metric_name_list = [
                        metric for metric in self._config.main_config.metrics
                    ]

                    train_scores = self.log(
                        train_loss / global_step,
                        train_loss_name,
                        metric_list,
                        metric_name_list,
                        train_logger,
                        train_log_values,
                        global_step,
                        append_text=self.train_config.append_text,
                    )

            if not os.path.exists(self.train_config.checkpoint.checkpoint_dir):
                os.makedirs(self.train_config.checkpoint.checkpoint_dir)

            torch.save(
                model.state_dict(),
                f"{self.train_config.checkpoint.checkpoint_dir}_{str(self.train_config.log.log_label)}"
                + "_"
                + str(epoch)
                + ".pth",
            )
            if epoch == max_epochs:
                print("\nEvaluating\n")
                val_scores = self.val(
                    model,
                    val_dataset,
                    criterion,
                    device,
                    global_step,
                    train_logger,
                    train_log_values,
                )
                save_flag = 0
                if self.train_config.save_on is not None:

                    train_loss_name = self.train_config.criterion.type
                    training_loss = train_loss / global_step

                    metric_list = [
                        metric(all_outputs.detach().cpu(), all_labels.cpu())
                        for metric in self.metrics
                    ]
                    metric_name_list = [
                        metric for metric in self._config.main_config.metrics
                    ]

                    train_scores = dict(
                        zip(
                            [train_loss_name,] + metric_name_list,
                            [training_loss,] + metric_list,
                        )
                    )

                    if self.train_config.save_on.desired == "min":
                        if val_scores[save_on_score] < best_score:
                            save_flag = 1
                            best_score = val_scores[save_on_score]
                            best_step = global_step
                    else:
                        if val_scores[save_on_score] > best_score:
                            save_flag = 1
                            best_score = val_scores[save_on_score]
                            best_step = global_step
                    if save_flag:
                        torch.save(
                            {
                                "model_state_dict": model,
                                "best_step": best_step,
                                "best_score": best_score,
                                "save_on_score": save_on_score,
                            },
                            self.train_config.save_on.best_path.format(self.log_label),
                        )

                        hparam_list = []
                        hparam_name_list = []
                        if self.train_config.log.values.hparams is not None:
                            for hparam in self.train_config.log.values.hparams:
                                hparam_list.append(
                                    get_item_in_config(self._config, hparam["path"])
                                )
                                hparam_name_list.append(hparam["name"])

                            val_keys, val_values = zip(*val_scores.items())
                            train_keys, train_values = zip(*train_scores.items())
                            val_keys = list(val_keys)
                            train_keys = list(train_keys)
                            val_values = list(val_values)
                            train_values = list(train_values)
                            for i, key in enumerate(val_keys):
                                val_keys[i] = "best_val_val_" + val_keys[i]
                            for i, key in enumerate(train_keys):
                                train_keys[i] = "best_val_train_" + train_keys[i]
                            train_logger.save_hyperparams(
                                hparam_list,
                                hparam_name_list,
                                train_values + val_values,
                                train_keys + val_keys,
                            )

                save_flag = 1
                if self.train_config.save_on is not None:

                    train_loss_name = self.train_config.criterion.type
                    training_loss = train_loss / global_step

                    metric_list = [
                        metric(all_outputs.detach().cpu(), all_labels.cpu())
                        for metric in self.metrics
                    ]
                    metric_name_list = [
                        metric for metric in self._config.main_config.metrics
                    ]

                    train_scores = dict(
                        zip(
                            [train_loss_name,] + metric_name_list,
                            [training_loss,] + metric_list,
                        )
                    )

                    torch.save(
                        {
                            "model_state_dict": model,
                            "final_step": global_step,
                            "final_score": train_scores[save_on_score],
                            "save_on_score": save_on_score,
                        },
                        self.train_config.save_on.final_path.format(self.log_label),
                    )

                    hparam_list = []
                    hparam_name_list = []
                    if self.train_config.log.values.hparams is not None:
                        for hparam in self.train_config.log.values.hparams:
                            hparam_list.append(
                                get_item_in_config(self._config, hparam["path"])
                            )
                            hparam_name_list.append(hparam["name"])

                        val_keys, val_values = zip(*val_scores.items())
                        train_keys, train_values = zip(*train_scores.items())
                        val_keys = list(val_keys)
                        train_keys = list(train_keys)
                        val_values = list(val_values)
                        train_values = list(train_values)
                        for i, key in enumerate(val_keys):
                            val_keys[i] = "final_val_" + val_keys[i]
                        for i, key in enumerate(train_keys):
                            train_keys[i] = "final_train_" + train_keys[i]
                        train_logger.save_hyperparams(
                            hparam_list,
                            hparam_name_list,
                            train_values + val_values,
                            train_keys + val_keys,
                        )

                print("\nLogging\n")
                train_loss_name = self.train_config.criterion.type
                metric_list = [
                    metric(outputs.detach().cpu(), labels.cpu())
                    for metric in self.metrics
                ]
                metric_name_list = [
                    metric for metric in self._config.main_config.metrics
                ]

                train_scores = self.log(
                    train_loss / global_step,
                    train_loss_name,
                    metric_list,
                    metric_name_list,
                    train_logger,
                    train_log_values,
                    global_step,
                    append_text=self.train_config.append_text,
                )

    ## Need to check if we want same loggers of different loggers for train and eval
    ## Evaluate

    def log(
        self,
        loss,
        loss_name,
        metric_list,
        metric_name_list,
        logger,
        log_values,
        global_step,
        append_text,
    ):

        return_dic = dict(zip([loss_name,] + metric_name_list, [loss,] + metric_list))

        loss_name = f"{append_text}_{self.log_label}_{loss_name}"
        if log_values["loss"]:
            logger.save_params(
                [loss],
                [loss_name],
                combine=True,
                combine_name="losses",
                global_step=global_step,
            )

        for i in range(len(metric_name_list)):
            metric_name_list[
                i
            ] = f"{append_text}_{self.log_label}_{metric_name_list[i]}"
        if log_values["metrics"]:
            logger.save_params(
                metric_list,
                metric_name_list,
                combine=True,
                combine_name="metrics",
                global_step=global_step,
            )
            # print(hparams_list)
            # print(hparam_name_list)

        # for k,v in dict(zip([loss_name],[loss])).items():
        #     print(f"{k}:{v}")
        # for k,v in dict(zip(metric_name_list,metric_list)).items():
        #     print(f"{k}:{v}")
        return return_dic

    def val(
        self,
        model,
        dataset,
        criterion,
        device,
        global_step,
        train_logger=None,
        train_log_values=None,
        log=True,
    ):
        append_text = self.val_config.append_text
        if train_logger is not None:
            val_logger = train_logger
        else:
            val_logger = Logger(**self.val_config.log.logger_params.as_dict())

        if train_log_values is not None:
            val_log_values = train_log_values
        else:
            val_log_values = self.val_config.log.values.as_dict()

        val_loader = DataLoader(
            dataset,
            **self.val_config.loader_params.as_dict(),
            collate_fn=dataset.custom_collate_fn,
        )

        all_outputs = torch.Tensor().to(device)
        all_labels = torch.FloatTensor().to(device)

        batch_size = self.val_config.loader_params.batch_size

        with torch.no_grad():
            val_loss = 0
            for j, batch in enumerate(val_loader):

                *inputs, labels = [
                    torch.tensor(value, device=device) for value in batch
                ]
                outputs = model(inputs)
                loss = criterion(torch.squeeze(outputs), labels)
                val_loss += loss.item()

                all_labels = torch.cat((all_labels, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)

            val_loss = val_loss / len(val_loader)

            val_loss_name = self.train_config.criterion.type
            metric_list = [
                metric(outputs.detach().cpu(), labels.cpu()) for metric in self.metrics
            ]
            metric_name_list = [metric for metric in self._config.main_config.metrics]
            return_dic = dict(
                zip([val_loss_name,] + metric_name_list, [loss,] + metric_list)
            )
            if log:
                val_scores = self.log(
                    val_loss,
                    val_loss_name,
                    metric_list,
                    metric_name_list,
                    val_logger,
                    val_log_values,
                    global_step,
                    append_text,
                )
            return return_dic
