import json
import os
from glob import glob
from typing import Dict, Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from managers.criteria import IntersectionOverUnionLoss
from managers.dataset import GaugeDataset
from misc.utils import display_predictions
from transforms.random_background import RandomBackground
from transforms.random_gauge_offset import RandomGaugeOffset
from transforms.resize_image import ResizeImage
from transforms.utils import Compose, ToTensor, Normalize


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
            for batch_idx, (images, targets) in enumerate(tqdm(self.__training_loader, desc=f'E-{epoch}', ncols=100)):
                loss, _ = self.__step(images, targets)
                self.__writer.add_scalar('Loss/Train', loss.item(), idx)
                idx += 1

            # Validation
            self.__model.eval()
            with torch.no_grad():
                losses = []
                validation_results = []

                for images, targets in tqdm(self.__validation_loader, desc=f'Validation', ncols=100):
                    loss, batch_results = self.__step(images, targets, evaluation=True, log_images=True)
                    losses.append(loss)
                    validation_results.append(batch_results)

                loss = sum(losses) / len(losses)
                validation_results = make_grid(torch.cat(validation_results))
                self.__writer.add_scalar('Loss/Validation', loss.item(), epoch)
                self.__writer.add_image('Validation results', validation_results, epoch)
                self.__save_checkpoint(epoch, {'loss': loss.item()})

                self.__scheduler.step(loss)

    def predict(self):
        self.__setup_datasets(evaluation=True)
        self.__model.eval()

        with torch.no_grad():
            for images, targets in tqdm(self.__validation_loader):
                images, targets = self.__send_data_to_device(images, targets)
                predictions = self.__model(images)
                display_predictions(images, predictions, targets['bbox'])

    def __step(self, images, targets, evaluation=False, log_images=False):
        images, targets = self.__send_data_to_device(images, targets)

        predictions = self.__model(images)
        loss = self.__criterion(predictions, targets['bbox'])

        if not evaluation:
            loss.backward()
            self.__optimizer.step()
            self.__optimizer.zero_grad()

        if log_images:
            results = display_predictions(images, predictions, targets=targets['bbox'], show=False)
        else:
            results = None

        return loss, results

    def __send_data_to_device(self, images, targets):
        images = images.to(self.__device)
        targets = {
            'bbox': torch.stack(targets['bbox']).transpose(0, 1).to(self.__device),
            'target': targets['target'].to(self.__device)
        }

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
        train_transforms = Compose([
            ResizeImage(resize_factor=0.25),
            RandomGaugeOffset(),
            RandomBackground('backgrounds'),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        validation_transforms = Compose([
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if not evaluation:
            self.__training_dataset = GaugeDataset(os.path.join(self.__root_path, 'training'), train_transforms)
            self.__training_loader = DataLoader(self.__training_dataset,
                                                batch_size=self.__batch_size, shuffle=True, num_workers=4)

            validation_path = os.path.join(self.__root_path, 'validation')
        else:
            validation_path = self.__root_path

        self.__validation_dataset = GaugeDataset(validation_path, validation_transforms)
        self.__validation_loader = DataLoader(self.__validation_dataset, num_workers=4)

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
        self.__scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer)
        self.__criterion = IntersectionOverUnionLoss()

    def __setup_logger(self):
        logs_base_path = 'logs'
        logs_prefix = 'Run_'

        if os.path.exists(logs_base_path):
            sub_logs = sorted(glob(os.path.join(logs_base_path, '*')),
                              key=lambda x: int(x.split(os.sep)[-1][len(logs_prefix):]))

            last_log_idx = 0 if len(sub_logs) == 0 else int(sub_logs[-1].split(os.sep)[-1][len(logs_prefix):])
        else:
            last_log_idx = 0

        self.__log_id = last_log_idx + 1
        self.__writer = SummaryWriter(log_dir=os.path.join(logs_base_path, f'{logs_prefix}{self.__log_id}'))
