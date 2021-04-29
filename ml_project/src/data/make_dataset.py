# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd

# from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


def load_raw_csv(filename: str) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(filename)
    except Exception as error:
        logger.error(
            f"Exception during loading raw dataset: {error.message if hasattr(error, 'message') else error}"
        )
        raise error
    logger.info("Raw dataset was successfully read")
    cleared_dataset = clear_raw_dataset(dataset)
    return cleared_dataset


def clear_raw_dataset(dataset: pd.DataFrame):
    dataset.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    return dataset[["label", "text"]]


def save_to_output(data: pd.DataFrame, output_filepath: str) -> None:
    data.to_csv(output_filepath, index=False)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("Processing raw data to final data set")
    loaded = load_raw_csv(input_filepath)
    save_to_output(loaded, output_filepath)
    logger.info(f"Raw data processed from {input_filepath} to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
