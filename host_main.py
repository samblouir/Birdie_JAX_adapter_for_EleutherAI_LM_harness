'''

Usage:
You must first add birdie.py in your eleutherlm/models directory.
This script uses Flask to launch a server that listens for a request from the EleutherLM Harness.

What this code does is:
	1. Launch the server
	2. Receive a request from the EleutherLM Harness
	3. This request contains the desired model tag and the inputs
	4. If we have not loaded a model:
		- Load the model
		- Store the model tag
	5. If we have already loaded a model, and it does not match the requested model tag:
		- Exit the program and let an outside script automatically call this again.
	6. Make a prediction using the model
	7. Send the prediction back to the EleutherLM Harness, which handles the rest from there.
	
'''

from flask import Flask, request, Response
import dill
import server_dest
import os
import sys
import traceback
import jax

pretty_filename = __file__.split("/")[-1]

file_dir = os.path.dirname(os.path.realpath(__file__))
tmp_host_has_started_flag_path = f"{file_dir}/_host_has_started_flag"


class ModelWrapper():
	def __init__(self, model):
		self.model = model
	
app = Flask(__name__)
@app.route('/data', methods=['POST'])
def data():
	global model
	global loaded_model_tag

	raw_data = request.data

	try:

		kwargs = dill.loads(raw_data)


		inputs = kwargs.get("inputs", [])
		if ("inputs" in kwargs) and (len(inputs) == 0):
			print(f'  Empty input! Sending back blank...')
			x = dill.dumps([])
			return Response(x, status=200, mimetype='application/octet-stream')
		
		if model is None:
			model = server_dest.get_model(**kwargs)
			loaded_model_tag = kwargs['model_tag']
			
		if loaded_model_tag != kwargs['model_tag']:
			print(f'  INFO: The requested model tag is not what is loaded. Requested: {kwargs["model_tag"]}, Loaded: {loaded_model_tag}')
			os._exit(1) # Dirty exit

			## Can't seem to get JAX to give up the model RAM without exiting...
			# del model
			# model = server_dest.get_model(**kwargs)
			# loaded_model_tag = kwargs['model_tag']

		# This function accepts the 
		rv = server_dest.predict(model, **kwargs,)

		x = dill.dumps(rv)

		return Response(x, status=200, mimetype='application/octet-stream')
	except Exception as e:

		print(f'\n'*5)
		print(f"#"*60)
		print(f"#"*60)
		print(f"  server_host.py:  Exception! e: {e}")
		print(f"#"*60)
		traceback.print_exc()
		print(f"#"*60)
		os._exit(1)
		return Response(f"Exception! e: {e}", status=400)
	finally:
		print(f"  server_host.py:  finally block: Done!")
		
##################################################
	




	
if __name__ == '__main__':
	model = None
	loaded_model_tag = None
	# cache_responses = False

	## Read in arguments
	arg_dict = {}
	for arg in sys.argv:
		if "=" in arg:
			key, val = arg.split("=")
			arg_dict[key] = int(val)
	# safe
	cuda_visible_devices = arg_dict.get('cuda_visible_devices', 0)
	port = int(arg_dict.get('port', '5000'))
	
	if (cuda_visible_devices != -1):
		os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
		os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
		os.environ['JAX_PLATFORM_NAME'] = 'gpu'
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = ""
		os.environ['JAX_PLATFORM_NAME'] = 'cpu'

	print(f"  Starting a server using these jax.devices(): {jax.devices()}")
	assert("cuda" in str(jax.devices()).lower())

	# Starts the server
	try:
		app.run(debug=False,  threaded=False, port=port,)
	except KeyboardInterrupt:
		print("Shutting down the server...")
















###################################################