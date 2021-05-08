import logging

import pandas as pd
import torch
from hydra.utils import to_absolute_path
from tqdm import tqdm

from src.config.config import Config
from src.constants.consts import APP_NAME, DATA
from src.data.make_dataset import load_processed_dataset
from src.models.utils.helpers import load_model, get_device
from src.utils.decorators import time_it

logger = logging.getLogger(APP_NAME)


def make_predictions(model, dataset, device):
    prediction_hist = []
    for sample in tqdm(dataset, total=len(dataset), desc="predict..."):
        data = torch.tensor(sample[DATA]).unsqueeze(dim=1)
        data = data.to(device)
        with torch.no_grad():
            predictions, _ = model(data, None)
        prediction = predictions[-1].detach().cpu()
        prediction_hist.append(prediction)
    predictions_list = torch.cat(prediction_hist, dim=0).numpy().tolist()
    return predictions_list


def save_predictions(predictions, filename):
    pd.DataFrame(dict(predictions=predictions)).to_csv(filename)


@time_it("predict_model, duration", logger.info)
def predict_model(cfg: Config):
    test_dataset = load_processed_dataset(to_absolute_path(cfg.test_dataset))
    vocab = torch.load(to_absolute_path(cfg.vocab_path))
    args = cfg.train

    device = get_device(args.gpu)
    model = load_model(args.model, vocab, args.dump_model)
    model.eval()
    model.to(device)

    predictions = make_predictions(model, test_dataset, device)
    save_predictions(predictions, cfg.predict.result_file)
