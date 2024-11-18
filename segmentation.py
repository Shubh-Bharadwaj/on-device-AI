model = Model.from_pretrained()
input_shape = (1, 3, 1024, 2048)
stats = summary(model, 
  input_size=input_shape, 
  col_names=["num_params", "mult_adds"]
)
print(stats)


from qai_hub_models.models.ffnet_40s import Model


low_res_input_shape = (1, 3, 512, 1024)

model = Model.from_pretrained()
stats = summary(model, 
  input_size=input_shape,
  col_names=["num_params", "mult_adds"]
)
print(stats)

from utils import get_ai_hub_api_token
ai_hub_api_token = get_ai_hub_api_token()

get_ipython().system('qai-hub configure --api_token $ai_hub_api_token')



get_ipython().run_line_magic('run', '-m qai_hub_models.models.ffnet_40s.demo')




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





get_ipython().run_line_magic('run', '-m qai_hub_models.models.ffnet_40s.export -- --device "$selected_device"')




get_ipython().run_line_magic('run', '-m qai_hub_models.models.ffnet_40s.demo -- --device "$selected_device" --on-device')


