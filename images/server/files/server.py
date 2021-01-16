"""
Demonstration server for Verge AI's Federated Learning technology.
"""
import base64
import os
import requests
import json
import torch
import numpy
from loguru import logger
from datetime import datetime

import vergeai
from pytorch_serde.serde import PyTorchSerializer
from pytorch_serde.serde import PyTorchDeserializer
from model import Net


def api_key() -> str:
    return os.environ["API_KEY"]


def initialize_api_key() -> None:
    vergeai.api_key = api_key()


if __name__ == '__main__':
    # Initialize API w/ API Key
    initialize_api_key()
    vergeai.initialize_logger("DEBUG")

    # Create project
    project_name = f"Demo {datetime.now()}"

    logger.info(f"Creating VergeAI project: \"{project_name}\"")
    project = vergeai.Project.create(
        project_name=project_name,
        project_description="test project, to verify Verge AI functionality")
    project_id = project.data["ID"]
    logger.success(f"Finished creating VergeAI project: \"{project_name}\"")

    # Load model from file
    logger.info("Loading demo model")
    with open("model.py", "rb") as f:
        code = base64.b64encode(f.read()).decode("utf-8")
    logger.success("Finished loading demo model")

    # Serialize experiment starting parameters
    logger.info("Serializing demo model")
    start_parameters = PyTorchSerializer().serialize(Net().state_dict())
    logger.success("Finished serializing demo model")

    logger.info("Your demo has been set up. Please return to the Getting Started guide for next steps")
    logger.info("You can exit by pressing Control+C (^C)")

    while True:
        user_input = input(">  ")

        if "start_experiment" in user_input:
            logger.info("Starting new experiment")
            experiment = vergeai.Experiment.create(
                project_id=project_id,
                experiment_name="Test Experiment",
                experiment_description="If you would like, you can add a custom experiment description here",
                runtime="CUSTOM",
                initialization_strategy="CUSTOMER_PROVIDED",
                data_collection="MINIMAL_RETAIN",
                aggregation_strategy="AVERAGE",
                ml_type="NN",
                code=code,
                learning_parameters=dict())
            experiment_id = experiment.data["ID"]
            logger.success(f"Created new experiment, with ID: {experiment_id}")

            logger.info("Submitting start model for experiment")
            vergeai.Experiment.submit_start_model(
                project_id=project_id,
                experiment_id=experiment_id,
                model=start_parameters,
                block=True)
            logger.success("Finished submitting start model for experiment")
