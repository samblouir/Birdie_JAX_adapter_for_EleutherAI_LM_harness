# JAX Adapter for EleutherAI LM Harness

This adapter, originally part of the Birdie codebase, facilitates the use of JAX models with the EleutherAI LM Harness. It is designed primarily for max-likelihood and multiple-choice tasks such as ARC and MMLU but can be extended to support additional tasks.

## Overview

The system is divided into two main components:

1. **Server**:
    - Deserializes the request.
    - Loads the requested model.
    - Runs the model on the inputs.
    - Serializes and sends the output back to the EleutherLM Harness.

2. **Custom model.py in EleutherLM Harness**:
    - Receives data and arguments from the EleutherLM Harness.
    - Sends a request to the server.
    - Processes the response.
    - Returns the response to the EleutherLM Harness.

## Notes

- Due to issues with JAX's official method for resetting allocated GPU VRAM, which caused segmentation faults, the server does not attempt to unload models dynamically. Instead, if a different model is requested, the server process exits. To manage this, the server script is automatically restarted in an outer loop for ease of use.

## Usage

### Setting Up

- Clone the EleutherAI LM Harness repository:
  ```bash
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git
  ```
- Move `birdie.py` to `lm-evaluation-harness/lm_eval/models/`.

### Running the Server

- Start the server using Flask. It listens for requests from the EleutherLM Harness and handles model loading and predictions accordingly.

### Workflow

1. Launch the server.
2. The server receives a request from the EleutherLM Harness.
3. The request contains the desired model tag and the inputs.
4. If the model has not been loaded:
    - Load the model.
    - Store the model tag.
5. If a different model is requested:
    - Exit the program.
    - An outside script automatically restarts the server to load the new model.
6. Make a prediction using the loaded model.
