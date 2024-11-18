
from qai_hub_models.models.ffnet_40s import Model as FFNet_40s


ffnet_40s = FFNet_40s.from_pretrained()




import torch
input_shape = (1, 3, 1024, 2048)
example_inputs = torch.rand(input_shape)




traced_model = torch.jit.trace(ffnet_40s, example_inputs)




traced_model


import qai_hub
import qai_hub_models

from utils import get_ai_hub_api_token
ai_hub_api_token = get_ai_hub_api_token()

get_ipython().system('qai-hub configure --api_token $ai_hub_api_token')




for device in qai_hub.get_devices():
    print(device.name)





devices = [
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy Tab S8",
    "Xiaomi 12",
    "Xiaomi 12 Pro",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
]

import random
selected_device = random.choice(devices)
print(selected_device)





device = qai_hub.Device(selected_device)


compile_job = qai_hub.submit_compile_job(
    model=traced_model,
    input_specs={"image": input_shape},
    device=device,
)



target_model = compile_job.get_target_model()




compile_options="--target_runtime tflite"
compile_options="--target_runtime onnx"
compile_options="--target_runtime qnn_lib_aarch64_android"

compile_job_expt = qai_hub.submit_compile_job(
    model=traced_model,
    input_specs={"image": input_shape},
    device=device,
    options=compile_options,
)




from qai_hub_models.utils.printing import print_profile_metrics_from_job


device = qai_hub.Device(selected_device)


profile_job = qai_hub.submit_profile_job(
    model=target_model,
    device=device,
)


profile_data = profile_job.download_profile()
print_profile_metrics_from_job(profile_job, profile_data)




profile_options="--compute_unit cpu"
profile_options="--compute_unit gpu"
profile_options="--compute_unit npu"

# Runs a performance profile on-device
profile_job_expt = qai_hub.submit_profile_job(
    model=target_model,
    device=device,
    options=profile_options,
)





sample_inputs = ffnet_40s.sample_inputs()
sample_inputs



torch_inputs = torch.Tensor(sample_inputs['image'][0])
torch_outputs = ffnet_40s(torch_inputs)
torch_outputs




inference_job = qai_hub.submit_inference_job(
        model=target_model,
        inputs=sample_inputs,
        device=device,
)





ondevice_outputs = inference_job.download_output_data()
ondevice_outputs['output_0']




from qai_hub_models.utils.printing import print_inference_metrics
print_inference_metrics(inference_job, ondevice_outputs, torch_outputs)



target_model = compile_job.get_target_model()
_ = target_model.download("FFNet_40s.tflite")



