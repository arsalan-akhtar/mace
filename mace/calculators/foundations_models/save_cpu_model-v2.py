#!/usr/bin/env python

import torch
import click
from mace.modules.models import ScaleShiftMACE  # Import the custom class
from mace.modules.blocks import LinearNodeEmbeddingBlock  # Import the new custom class

@click.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to the GPU-enabled model file.')
@click.option('--output-path', required=True, type=click.Path(), help='Path to save the CPU-only model.')
def save_cpu_model(model_path, output_path):
    """
    This script loads a PyTorch model saved with GPU (CUDA) support, maps it to CPU, and saves it as a CPU-only model.
    """
    try:
        # Add the 'set', 'ScaleShiftMACE', and 'LinearNodeEmbeddingBlock' to the safe globals list
        torch.serialization.add_safe_globals([set, ScaleShiftMACE, LinearNodeEmbeddingBlock])

        # Load the model with map_location set to CPU and weights_only=True for safety
        click.echo(f"Loading the model from {model_path} onto CPU...")
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        # Save the model in CPU format
        click.echo(f"Saving the CPU-only model to {output_path}...")
        torch.save(model, output_path)

        click.echo("Model successfully saved as CPU-only!")
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    save_cpu_model()

