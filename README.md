[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py

# task 2_5:
## simple dataset
parameters:
* lr=0.1
* epochs=500
* hidden_layer_size=2
* accuracy=100%
* time_per_epoch=0.041s
* num_points=50

![image](https://github.com/user-attachments/assets/9d012b9e-0d3b-43b8-9ec1-01848c08c688)

## diag dataset
parameters:
* lr=0.5
* epochs=500
* hidden_layer_size=4
* accuracy=100%
* time_per_epoch=0.083s
* num_points=50

![{EB5CAA9E-9429-46F9-B84D-0EC92DEEB816}](https://github.com/user-attachments/assets/c7ce17cf-e8e5-4f00-b6e4-f5b125c8a57e)

## split dataset
parameters:
* lr=0.1
* epochs=1000
* hidden_layer_size=8
* accuracy=98%
* time_per_epoch=0.209s
* num_points=50

![image](https://github.com/user-attachments/assets/9365b19f-a5d8-411c-97c6-87937fce2552)

## xor dataset
parameters:
* lr=0.1
* epochs=1000
* hidden_layer_size=16
* accuracy=100%
* num_points=50
* time_per_epoch=0.646s

![image](https://github.com/user-attachments/assets/58f85f5c-e083-4652-9c75-cd8467367436)
