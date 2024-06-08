# SNN_frame_detector
These files are Wi-Fi frame detectors using Spiking Neural Networks with memristive synapses.
The Python files demand BindsNET to operate.
Furthermor, they require our add-on packages to apply memristive synapses for weight updates.
Please install BindsNET and use our detector files.

# How to use additional package
We can't upload our additional packages because they are prohibitted to be released.
If you need the package and more datasets, please contact me.
hetzer44@naver.com
Then, I will send the package and datasets.
For using the package, just put the package into the bindsnet folder in the python library folder (site-packages)

# Memristive characteristic implemetation
We develop an additional package to implement memristive characteristics (nonlinear weight updates) in BindsNET.
To update synaptic weights with memristive charactersitic we use the equations in the paper named 'Resistive memory device requirements for a neural algorithm accelerator' (https://ieeexplore.ieee.org/document/7727298)
Our frame detector uses STDP learning rule and updates synaptic weights according to the following codes

``` python
  self.connection.w[i, k.item()] += b * (gmax[i, k.item()] - gmin[i, k.item()]) / 256 # LTP (linear)
  
  self.connection.w[i, k.item()] -= (gmax[i, k.item()] - gmin[i, k.item()]) / 256 # LTD (linear)
  
  self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256)) # LTP (nonlinear)
  
  self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] + g1ltd[i, k.item()] - gmax[i, k.item()]) * (1 - np.exp(vltd / 256)) # LTD (nonlinear)
```

To control nonlinearity (memristive characteristic), you should change the three parameters: vLTP, vLTD, and beta (b).
We add the three parameters in our mani Python file, so you can easily find and change them.
If you have questions, please email me.

# Dataset
We generate datasets consisting of Wi-Fi preamble samples and noise samples.
We compress a sample dataset and upload it at this repository.
If you need additional datasets, please email me.
I will provide you with the additional datasets as soon as possible.

# Paper
https://www.sciencedirect.com/science/article/pii/S0140366423002025
