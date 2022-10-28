#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
# import imageio

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats as sts
from scipy import stats
# plt.style.use('ggplot')   # Правильный стиль графиков   
plt.style.use('seaborn-whitegrid')

import seaborn as sns
sns.set_palette(['#00A99D', '#F5CA0C', '#B6129F', '#76620C', '#095C57'])

deep_grey = '#3B4856' # основной тёмный / холодный цвет
sky_blue = '#348FEA'
marine_green = '#4CB9C0'
grass_green = '#97C804'
medium_yellow = '#FFC100' # основной светлый / тёплый цвет
sicilian_orange = '#E06A27'
fuchsia_pink = '#C81D6B'
saturated_violet = '#5002A7'
navy_blue = '#292183'
cool_white = '#F5FBFF'

# from matplotlib import rc
# rc('text', usetex=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def mse(w):
    return 0.5*((2 - w)**2 + (3 - 2*w)**2)

def mse_grad(x, y, w):
    return -2 * x * (y - w * x)

grad_full = lambda w: mse_grad(1, 2, w) + mse_grad(2,3,w)

def line(w, w0):
    return grad_full(w)*(w - w0) + mse(w0)

w = np.linspace(-0.2, 3.4, 1000)


# In[3]:


x_dots = np.array([1.6])
y_dots = mse(x_dots)

plt.plot(w, mse(w), lw=2, color='black'); 

plt.scatter(x_dots, y_dots, color=fuchsia_pink, lw=6)

# w_dots = np.array([0, 0.8, 1.2, 1.4])
w_dots = np.array([0, 0.4, 1.28, 1.424, 1.4848])

plt.scatter(w_dots, mse(w_dots), color=sky_blue, lw=3)
# for ww in w_dots:
#     plt.vlines(x = ww, ymin = 0, ymax = mse(ww), linestyle='--', lw=2, color=sky_blue)

plt.xlabel('$w$', fontsize=16, color='black')
plt.ylabel(r'$MSE(w)$', fontsize=16, color='black')
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)


# In[ ]:





# In[4]:


for n in range(1,10):
    print(n, 2**(n + 1) - 1)


# In[ ]:





# In[ ]:




