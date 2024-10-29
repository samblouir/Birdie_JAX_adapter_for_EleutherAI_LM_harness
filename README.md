# JAX Adapter for EleutherAI LM Harness

This adapter, originally part of the Birdie codebase, allows for the use of JAX (and other framework) models with the EleutherAI LM Harness. It is designed originally for max-likelihood/multiple-choice tasks, such as ARC and MMLU, but will be extended to support generative tasks.

## Overview

The system is divided into three main components:

1. **API.py**
   - You just need to create two functions: one that loads your model and returns the logits from the vocabulary head. A second that supports tokenization from a string or list of strings. This is more clearly shown in the file itself.
1. **Server**:
    - Deserializes the request.
    - Loads the requested model.
    - Runs the model on the inputs.
    - Serializes and sends the output back to the EleutherLM Harness.

2. **Custom model.py in EleutherLM Harness**:
   - You can install this by copying `birdie.py` to `lm-evaluation-harness/lm_eval/models/`.

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

- Start the server using
  ```bash
  python host_main.py
  ```
It listens for requests from the EleutherLM Harness and handles model loading and predictions accordingly.

### Starting Tasks
Here is a real example of how I used this harness in Birdie
```bash
    # Comma-seperated list of tasks to run
    tasks="boolq"
    # Index of the GPU(s) to use, on your machine
    gpu_id=0
    # Number of fewshot examples to use (Note: The Eleuther harness doesn't support this for all tasks)
    num_fewshot=3
    results_dir="/home/sam/birdie_eleuther_results"
    cache_dir="/home/sam/birdie_eleuther_cache"
	model_tag="attention_trained_using_birdie_14B"


    # Json output path. This will save your results to a json file
    # NOTE: If you do too many tasks at once, this may create a file that is too long for your OS!
    output_path=${results_dir}/${tasks}_fewshot:${num_fewshot}.json

    # Ensures the results directory exists
    mkdir -p ${results_dir}/${model_tag} 
     # Place your model args using comma-seperated values
	model_args="--model_args model_tag=${model_tag},port=5000" 
	cache_args="--device cuda:${gpu_id} --use_cache ${cache_dir} 
	fewshot_args="--num_fewshot ${num_fewshot}"

    # Runs it! This will start Eleuther's harness, and thanks to your custom model, the requests will be forwarded to our server running in host_main.py, which will load the model, tokenize the inputs, and return the final losses to the harness.
	python /home/sam/lm-evaluation-harness/lm_eval/__main__.py --model birdie ${model_args} ${fewshot_args} --tasks ${tasks}  --output_path ${output_args} ${cache_args}

    # That's all!
```


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
