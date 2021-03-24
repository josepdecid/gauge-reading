import json
import os
from glob import glob
from typing import Dict, Optional

import torch
from torch import optim, nn
from torch.nn import MSELoss
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
    def __init__(self, bbox_model: nn.Module, value_model: nn.Module, root_path: str, batch_size: int,
                 bbox_checkpoint: Optional[str] = None, value_checkpoint: Optional[str] = None):

        self.__bbox_model = bbox_model
        self.__value_model = value_model

        self.__root_path = root_path
        self.__batch_size = batch_size
        self.__metrics = {}

        self.__setup_model(bbox_checkpoint, value_checkpoint)
        self.__setup_logger()

    def train(self, epochs: int):
        self.__setup_datasets()

        idx = 0
        for epoch in range(1, epochs):
            # Training
            self.__bbox_model.train()
            self.__value_model.train()
            for batch_idx, (images, targets) in enumerate(tqdm(self.__training_loader, desc=f'E-{epoch}', ncols=100)):
                bbox_loss, value_loss, _ = self.__step(images, targets)
                self.__writer.add_scalar('IntersectionOverUnion/Train', -bbox_loss.item(), idx)
                self.__writer.add_scalar('MeanSquareError/Train', value_loss.item(), idx)
                idx += 1

            # Validation
            self.__bbox_model.eval()
            self.__value_model.eval()
            with torch.no_grad():
                bbox_losses = []
                value_losses = []
                validation_results = []

                for images, targets in tqdm(self.__validation_loader, desc=f'Validation', ncols=100):
                    bbox_loss, value_loss, batch_results = self.__step(images, targets,
                                                                       evaluation=True, log_images=True)
                    bbox_losses.append(bbox_loss)
                    value_losses.append(value_loss)
                    validation_results.append(batch_results)

                bbox_loss = sum(bbox_losses) / len(bbox_losses)
                value_loss = sum(value_losses) / len(value_losses)

                self.__writer.add_scalar('IntersectionOverUnion/Validation', -bbox_loss.item(), epoch)
                self.__writer.add_scalar('MeanSquareError/Validation', value_loss.item(), epoch)

                validation_results = make_grid(torch.cat(validation_results))
                self.__writer.add_image('Validation results', validation_results, epoch)

                self.__save_checkpoint(epoch, {'bbox_loss': bbox_loss.item(), 'value_loss': value_loss.item()})

                self.__bbox_scheduler.step(bbox_loss)
                self.__value_scheduler.step(value_loss)

    def predict(self):
        self.__setup_datasets(evaluation=True)
        self.__bbox_model.eval()
        self.__value_model.eval()

        with torch.no_grad():
            for images, targets in tqdm(self.__validation_loader):
                images, targets = self.__send_data_to_device(images, targets)
                bbox_predictions = self.__bbox_model(images)
                value_predictions = self.__value_model(images)

                display_predictions(images, bbox_predictions, targets, value_predictions)

    def __step(self, images, targets, evaluation=False, log_images=False):
        images, targets = self.__send_data_to_device(images, targets)

        # Bounding box
        predictions = self.__bbox_model(images)
        bbox_loss = self.__bbox_criterion(predictions, targets['bbox'])
        if not evaluation:
            bbox_loss.backward()
            self.__bbox_optimizer.step()
            self.__bbox_optimizer.zero_grad()

        if log_images:
            results = display_predictions(images, predictions, targets=targets, show=False)
        else:
            results = None

        # Gauge value reading
        predictions = self.__value_model(images)
        value_loss = self.__value_criterion(predictions.view(-1), targets['target'])
        if not evaluation:
            value_loss.backward()
            self.__value_optimizer.step()
            self.__value_optimizer.zero_grad()

        return bbox_loss, value_loss, results

    def __send_data_to_device(self, images, targets):
        images = images.to(self.__device)
        targets = {
            'bbox': torch.stack(targets['bbox']).transpose(0, 1).to(self.__device),
            'target': targets['target'].type(torch.FloatTensor).to(self.__device)
        }

        return images, targets

    def __save_checkpoint(self, epoch: int, loss_metrics: Dict[str, float]):
        base_path = os.path.join('checkpoints', str(self.__log_id))
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        checkpoint_path = os.path.join(base_path, f'{epoch}_bbox.pt')
        torch.save(self.__bbox_model.state_dict(), checkpoint_path)

        checkpoint_path = os.path.join(base_path, f'{epoch}_value.pt')
        torch.save(self.__value_model.state_dict(), checkpoint_path)

        metrics_path = os.path.join(base_path, 'metrics.json')
        self.__metrics[epoch] = loss_metrics

        with open(metrics_path, mode='w') as json_file:
            json.dump(self.__metrics, json_file)

    def __setup_datasets(self, evaluation=False):
        train_transforms = Compose([
            ResizeImage(resize_factor=0.25),
            RandomGaugeOffset(),
            RandomBackground('backgrounds'),
            # TODO: Other traditional transformations (also modifying and calculating the bbox)
            # e.g. RandomResizedCrop, RandomPerspective, RandomHorizontalFlip...
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

    def __setup_model(self, bbox_checkpoint: Optional[str], value_checkpoint: Optional[str]):
        if torch.cuda.is_available():
            print('Using GPU')
            self.__device = torch.device('cuda')
        else:
            print('No available GPUs')
            self.__device = torch.device('cpu')

        # Send bounding box prediction model to device and load checkpoint if required
        self.__bbox_model = self.__bbox_model.to(self.__device)
        if bbox_checkpoint is not None:
            self.__bbox_model.load_state_dict(torch.load(bbox_checkpoint))

        self.__bbox_optimizer = optim.Adam(self.__bbox_model.parameters())
        self.__bbox_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__bbox_optimizer)
        self.__bbox_criterion = IntersectionOverUnionLoss()

        # Send regression gauge value model to device and load checkpoint if required
        self.__value_model = self.__value_model.to(self.__device)
        if value_checkpoint is not None:
            self.__value_model.load_state_dict(torch.load(value_checkpoint))

        self.__value_optimizer = optim.Adam(self.__value_model.parameters())
        self.__value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__value_optimizer)
        self.__value_criterion = MSELoss()

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
        print(f'Running experiment {self.__log_id}')
