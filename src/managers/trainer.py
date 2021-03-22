import json
import os
from glob import glob
from typing import Dict, Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from managers.dataset import GaugeDataset
from transforms.random_gauge_offset import RandomGaugeOffset
from transforms.utils import Compose
from misc.utils import collate_fn, display_predictions


class Trainer:
    def __init__(self, model: nn.Module, root_path: str, batch_size: int, checkpoint_path: Optional[str] = None):
        self.__model = model
        self.__root_path = root_path
        self.__batch_size = batch_size
        self.__metrics = {}

        self.__setup_model(checkpoint_path)
        self.__setup_logger()

    def train(self, epochs: int):
        self.__setup_datasets()

        idx = 0
        for epoch in range(1, epochs):
            # Training
            self.__model.train()
            for images, targets in tqdm(self.__training_loader):
                images, targets = self.__send_data_to_device(images, targets)

                self.__optimizer.zero_grad()
                loss_dict = self.__model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.__optimizer.step()

                self.__writer.add_scalar('Loss/Train', losses.item(), idx)
                idx += 1

            # Validation
            # self.__model.eval()
            with torch.no_grad():
                losses = []
                for images, targets in tqdm(self.__validation_loader):
                    images, targets = self.__send_data_to_device(images, targets)

                    loss_dict = self.__model(images, targets)
                    losses.append(loss_dict)

                loss_metrics = {}
                for k in losses[0].keys():
                    key_loss = sum(map(lambda x: x[k].item(), losses)) / len(losses)
                    self.__writer.add_scalar(f'Loss/Validation/{k}', key_loss, epoch)
                    loss_metrics[k] = sum(map(lambda x: x[k].item(), losses)) / len(losses)

                self.__save_checkpoint(epoch, loss_metrics)

    def predict(self):
        self.__setup_datasets(evaluation=True)
        self.__model.eval()

        with torch.no_grad():
            for images, targets in tqdm(self.__validation_loader):
                images, targets = self.__send_data_to_device(images, targets)
                predictions = self.__model(images, targets)
                display_predictions(images, predictions)

    def __send_data_to_device(self, images, targets):
        images = list(image.to(self.__device) for image in images)
        targets = [{
            k: v.to(self.__device) if torch.is_tensor(v) else v
            for k, v in t.items()
        } for t in targets]

        return images, targets

    def __save_checkpoint(self, epoch: int, loss_metrics: Dict[str, float]):
        base_path = os.path.join('checkpoints', str(self.__log_id))
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        checkpoint_path = os.path.join(base_path, f'{epoch}.pt')
        torch.save(self.__model.state_dict(), checkpoint_path)

        metrics_path = os.path.join(base_path, 'metrics.json')
        self.__metrics[epoch] = loss_metrics

        with open(metrics_path, mode='w') as json_file:
            json.dump(self.__metrics, json_file)

    def __setup_datasets(self, evaluation=False):
        transforms = Compose([RandomGaugeOffset()])

        if not evaluation:
            self.__training_dataset = GaugeDataset(os.path.join(self.__root_path, 'training'), transforms)
            self.__training_loader = DataLoader(self.__training_dataset, collate_fn=collate_fn,
                                                batch_size=self.__batch_size, shuffle=True, num_workers=4)

            validation_path = os.path.join(self.__root_path, 'validation')
        else:
            validation_path = self.__root_path

        self.__validation_dataset = GaugeDataset(validation_path, transforms)
        self.__validation_loader = DataLoader(self.__validation_dataset, collate_fn=collate_fn,
                                              batch_size=self.__batch_size, num_workers=4)

    def __setup_model(self, checkpoint_path: Optional[str]):
        if torch.cuda.is_available():
            print('Using GPU')
            self.__device = torch.device('cuda')
        else:
            print('No available GPUs')
            self.__device = torch.device('cpu')

        self.__model = self.__model.to(self.__device)
        if checkpoint_path is not None:
            self.__model.load_state_dict(torch.load(checkpoint_path))

        self.__optimizer = optim.Adam(self.__model.parameters())
        self.__criterion = nn.MSELoss()

    def __setup_logger(self):
        logs_base_path = 'logs'
        logs_prefix = 'Run_'

        if os.path.exists(logs_base_path):
            sub_logs = sorted(glob(os.path.join(logs_base_path, '*')))
            last_log_idx = 0 if len(sub_logs) == 0 else int(sub_logs[-1].split(os.sep)[-1][len(logs_prefix):])
        else:
            last_log_idx = 0

        self.__log_id = last_log_idx + 1
        self.__writer = SummaryWriter(log_dir=os.path.join(logs_base_path, f'{logs_prefix}{self.__log_id}'))
