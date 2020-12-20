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


@configmapper.map("trainers","forty")
class FortyTrainer:
    def __init__(self,config):
        self._config = config
        self.metrics = [configmapper.get_object('metrics',metric) for metric in self._config.main_config.metrics]
        self.train_config = self._config.train
        self.val_config = self._config.val

    def train(self, model, train_dataset, val_dataset=None):
        device = torch.device(self._config.main_config.device.name)
        model.to(device)
        optim_params = self.train_config.optimizer.params
        if(optim_params):
            optimizer = configmapper.get_object('optimizers',self.train_config.optimizer.type)(model.parameters(),**optim_params.as_dict())
        else:
            optimizer = configmapper.get_object('optimizers',self.train_config.optimizer.type)(model.parameters())

        if(self.train_config.scheduler is not None):
            scheduler_params = self.train_config.scheduler.params
            if(scheduler_params):
                scheduler = configmapper.get_object('schedulers',self.train_config.scheduler.type)(optimizer,**scheduler_params.as_dict())
            else:
                scheduler = configmapper.get_object('schedulers',self.train_config.scheduler.type)(optimizer)

        criterion_params = self.train_config.criterion.params
        if(criterion_params):
            criterion = configmapper.get_object('losses',self.train_config.criterion.type)(**criterion_params.as_dict())
        else:
            criterion = configmapper.get_object('losses',self.train_config.criterion.type)()

        train_loader = DataLoader(train_dataset,**self.train_config.loader_params.as_dict())
        # train_logger = Logger(**self.train_config.log.logger_params.as_dict())

        max_epochs = self.train_config.max_epochs
        batch_size = self.train_config.loader_params.batch_size

        log_interval = self.train_config.log.log_interval
        train_logger = Logger(**self.train_config.log.logger_params.as_dict())
        train_log_values = self.train_config.log.values.as_dict()

        val_interval = self.train_config.val_interval


        break_all=False

        print('\nTraining\n')
        # print(max_steps)

        global_step = 0
        for epoch in range(1, max_epochs + 1):
            print("Epoch: {}/{}, Global Step: {}".format(epoch,max_epochs, global_step))
            train_loss = 0
            val_loss = 0

            all_labels = torch.FloatTensor().to(device)
            all_outputs = torch.Tensor().to(device)

            pbar = tqdm(total=math.ceil(len(train_dataset) / batch_size))
            pbar.set_description("Epoch " + str(epoch))

            val_counter = 0

            for step,batch in enumerate(train_loader):
                optimizer.zero_grad()
                *inputs, labels = [value.to(device) for value in batch]
                labels = labels.float()
                outputs = model(*inputs)
                loss = criterion(torch.squeeze(outputs),labels)
                loss.backward()

                all_labels = torch.cat((all_labels,labels),0)
                all_outputs = torch.cat((all_outputs,outputs),0)

                train_loss+=loss.item()
                optimizer.step()

                if (self.train_config.scheduler is not None):
                    scheduler.step(epoch + i/len(train_loader))

                # print(train_loss)
                # print(step+1)

                pbar.set_postfix_str(f"Train Loss: {train_loss /(step+1)}")
                pbar.update(1)

                global_step+=1

#Need to check if we want global_step or local_step
                if(val_dataset is not None and (global_step-1)%val_interval==0):
                    print("\nEvaluating\n")
                    self.val(model,val_dataset,criterion,device,global_step,train_logger,train_log_values)

                if((global_step-1)%log_interval==0):
                    print("\nLogging\n")
                    self.log(train_loss/global_step,f"Train {self.train_config.criterion.type}",all_labels,all_outputs,train_logger,train_log_values,global_step)
            if not os.path.exists(self.train_config.checkpoint.checkpoint_dir):
                os.makedirs(self.train_config.checkpoint.checkpoint_dir)

            torch.save(
                model.state_dict(),
                f"{self.train_config.checkpoint.checkpoint_dir}_{str(self.train_config.log.log_label)}"
                + "_"
                + str(epoch)
                + ".pth",
            )

## Need to check if we want same loggers of different loggers for train and eval
## Evaluate
    def log(self,loss,loss_name,labels,outputs,logger,log_values,global_step, append_text="Train "):
        if (log_values['loss']):
            logger.save_params([loss],[loss_name],global_step=global_step)

        metric_list =[metric(outputs.detach().cpu(),labels.cpu()) for metric in self.metrics]
        metric_name_list= [append_text+metric for metric in self._config.main_config.metrics]
        if(log_values['metrics']):
            logger.save_params(metric_list,metric_name_list,combine=True,combine_name='metrics',global_step=global_step)
        # for k,v in dict(zip([loss_name],[loss])).items():
        #     print(f"{k}:{v}")
        # for k,v in dict(zip(metric_name_list,metric_list)).items():
        #     print(f"{k}:{v}")

    def val(self,model,dataset,criterion,device,global_step,train_logger,train_log_values):
        # val_logger = Logger(**self.val_config.log.logger_params.as_dict())
        val_logger = train_logger
        val_loader = DataLoader(dataset,**self.val_config.loader_params.as_dict())
        # val_log_values = self.val_config.log.values.as_dict()
        val_log_values = train_log_values

        all_outputs = torch.Tensor().to(device)
        all_labels = torch.FloatTensor().to(device)

        batch_size = self.val_config.loader_params.batch_size

        # print("\nEvaluating\n")
        # print(batch_size)
        # print(len(val_loader))
        # print(len(dataset))
        
        with torch.no_grad():
            val_loss = 0
            for j,batch in enumerate(val_loader):

                *inputs, labels = [value.to(device) for value in batch]
                outputs = model(*inputs)
                loss = criterion(torch.squeeze(outputs),labels)
                val_loss+=loss.item()

                all_labels = torch.cat((all_labels,labels),0)
                all_outputs = torch.cat((all_outputs,outputs),0)



            val_loss = val_loss/len(val_loader)
            self.log(val_loss,f"Val {self.train_config.criterion.type}",all_labels,all_outputs,val_logger,val_log_values,global_step, append_text="Val ")
