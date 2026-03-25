# -*- coding: utf-8 -*-
"""
Created on Thur Apr 17 13:00:00 2025

@author: Gregory A. Greene

The hourlyFFMC_lawson function.
This code was translated to Python from the C++ code in the WISE_FWI_Module.
"""
import numpy as np
from typing import Union

# Morning hour lookup tables for low (L) RH class
L = [
    [9999.0, 17.5, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 72.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0,
     83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 100.9,
     101.0],
    [600.0, 48.3, 49.4, 51.1, 53.5, 55.1, 56.9, 59.1, 61.7, 62.9, 64.1, 64.8, 65.5, 66.2, 66.9, 67.7, 68.5, 69.4, 70.2,
     71.1, 72.1, 73.1, 74.1, 75.2, 76.3, 77.5, 78.7, 80.0, 81.3, 82.7, 84.1, 85.7, 87.2, 88.8, 90.4, 91.9, 93.2, 93.8,
     93.8],
    [700.0, 50.7, 52.1, 53.9, 56.3, 57.9, 59.7, 61.8, 64.3, 65.4, 66.6, 67.2, 67.9, 68.6, 69.3, 70.0, 70.7, 71.5, 72.3,
     73.2, 74.0, 75.0, 75.9, 76.9, 77.9, 79.0, 80.2, 81.4, 82.6, 83.9, 85.2, 86.6, 88.1, 89.6, 91.1, 92.6, 93.9, 94.5,
     94.5],
    [800.0, 53.3, 54.9, 56.8, 59.3, 60.9, 62.6, 64.7, 67.0, 68.1, 69.2, 69.8, 70.4, 71.0, 71.6, 72.3, 73.0, 73.7, 74.5,
     75.3, 76.1, 76.9, 77.8, 78.7, 79.7, 80.6, 81.7, 82.8, 83.9, 85.1, 86.3, 87.7, 89.0, 90.4, 91.9, 93.3, 94.6, 95.3,
     95.3],
    [900.0, 59.6, 60.7, 62.2, 64.4, 65.7, 67.3, 69.1, 71.2, 72.1, 73.2, 73.7, 74.2, 74.8, 75.4, 76.0, 76.7, 77.3, 78.0,
     78.7, 79.5, 80.3, 81.1, 81.9, 82.8, 83.7, 84.7, 85.7, 86.7, 87.8, 89.0, 90.1, 91.4, 92.6, 93.9, 95.2, 96.3, 96.8,
     96.8],
    [1000.0, 66.8, 67.2, 68.2, 69.9, 70.9, 72.2, 73.8, 75.6, 76.5, 77.4, 77.9, 78.4, 78.9, 79.4, 80.0, 80.5, 81.1, 81.8,
     82.4, 83.1, 83.8, 84.5, 85.3, 86.1, 86.9, 87.8, 88.7, 89.7, 90.6, 91.7, 92.7, 93.8, 94.9, 96.0, 97.1, 97.9, 98.4,
     98.4],
    [1100.0, 74.5, 74.5, 74.9, 75.9, 76.6, 77.6, 78.8, 80.3, 81.0, 81.9, 82.4, 83.0, 83.6, 84.1, 84.7, 85.2, 85.8, 86.3,
     86.9, 87.4, 88.0, 88.5, 89.0, 89.6, 90.1, 90.6, 91.1, 91.6, 92.1, 92.6, 93.1, 93.8, 94.9, 96.0, 97.1, 97.9, 98.4,
     98.4],
    [1159.0, 83.0, 82.5, 82.3, 82.4, 82.7, 83.2, 84.1, 85.2, 85.8, 86.5, 86.8, 87.2, 87.6, 87.9, 88.2, 88.6, 88.9, 89.2,
     89.6, 89.9, 90.2, 90.5, 90.9, 91.2, 91.5, 91.8, 92.1, 92.4, 92.7, 93.0, 93.3, 93.8, 94.9, 96.0, 97.1, 97.9, 98.4,
     98.4],
    [1200.0, 83.0, 82.5, 82.3, 82.4, 82.7, 83.2, 84.1, 85.2, 85.8, 86.5, 86.8, 87.2, 87.6, 87.9, 88.2, 88.6, 88.9, 89.2,
     89.6, 89.9, 90.2, 90.5, 90.9, 91.2, 91.5, 91.8, 92.1, 92.4, 92.7, 93.0, 93.3, 93.8, 94.9, 96.0, 97.1, 97.9, 98.4,
     98.4],
]
# Morning hour lookup tables for medium RH class
M = [
    [9999.0, 17.5, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 72.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0,
     83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 100.9,
     101.0],
    [600.0, 34.8, 39.2, 43.2, 47.6, 50.0, 52.6, 55.4, 58.4, 59.7, 61.1, 61.8, 62.5, 63.3, 64.0, 64.8, 65.6, 66.4, 67.2,
     68.1, 68.9, 69.8, 70.8, 71.7, 72.7, 73.8, 74.8, 75.9, 77.1, 78.3, 79.5, 80.8, 82.2, 83.6, 85.0, 86.5, 88.0, 89.1,
     89.1],
    [700.0, 36.3, 40.5, 44.3, 48.7, 51.2, 53.8, 56.7, 59.9, 61.3, 62.7, 63.4, 64.2, 64.9, 65.7, 66.5, 67.4, 68.2, 69.1,
     70.0, 70.9, 71.9, 72.8, 73.9, 74.9, 75.9, 77.0, 78.2, 79.3, 80.5, 81.8, 83.1, 84.4, 85.7, 87.0, 88.3, 89.5, 90.2,
     90.2],
    [800.0, 37.8, 41.7, 45.5, 49.8, 52.3, 55.1, 58.1, 61.4, 62.8, 64.3, 65.1, 65.9, 66.7, 67.5, 68.4, 69.3, 70.1, 71.1,
     72.0, 73.0, 74.0, 75.0, 76.0, 77.1, 78.2, 79.3, 80.5, 81.7, 82.9, 84.1, 85.4, 86.6, 87.9, 89.1, 90.2, 91.2, 91.6,
     91.6],
    [900.0, 44.6, 48.2, 51.6, 55.6, 57.8, 60.3, 63.0, 66.0, 67.3, 68.6, 69.3, 70.1, 70.8, 71.6, 72.3, 73.1, 73.9, 74.8,
     75.6, 76.5, 77.4, 78.3, 79.3, 80.3, 81.3, 82.3, 83.4, 84.5, 85.7, 86.8, 88.0, 89.2, 90.5, 91.7, 92.8, 93.8, 94.4,
     94.4],
    [1000.0, 52.5, 55.5, 58.5, 61.9, 63.9, 66.0, 68.4, 71.0, 72.1, 73.3, 73.9, 74.5, 75.2, 75.9, 76.5, 77.2, 77.9, 78.7,
     79.4, 80.2, 81.0, 81.9, 82.7, 83.6, 84.5, 85.5, 86.5, 87.5, 88.5, 89.6, 90.8, 91.9, 93.1, 94.3, 95.5, 96.7, 97.3,
     97.3],
    [1100.0, 61.6, 64.0, 66.3, 69.0, 70.6, 72.3, 74.2, 76.4, 77.3, 78.3, 79.0, 79.6, 80.3, 80.9, 81.5, 82.2, 82.8, 83.4,
     84.0, 84.6, 85.3, 85.9, 86.5, 87.1, 87.7, 88.3, 88.9, 89.4, 90.0, 90.6, 91.2, 91.9, 93.1, 94.3, 95.5, 96.7, 97.3,
     97.3],
    [1159.0, 72.1, 73.5, 75.0, 76.9, 77.9, 79.2, 80.6, 82.2, 82.9, 83.6, 84.0, 84.4, 84.8, 85.2, 85.6, 86.0, 86.4, 86.7,
     87.1, 87.5, 87.9, 88.2, 88.6, 88.9, 89.3, 89.7, 90.0, 90.3, 90.7, 91.0, 91.4, 91.9, 93.1, 94.3, 95.5, 96.7, 97.3,
     97.3],
    [1200.0, 72.1, 73.5, 75.0, 76.9, 77.9, 79.2, 80.6, 82.2, 82.9, 83.6, 84.0, 84.4, 84.8, 85.2, 85.6, 86.0, 86.4, 86.7,
     87.1, 87.5, 87.9, 88.2, 88.6, 88.9, 89.3, 89.7, 90.0, 90.3, 90.7, 91.0, 91.4, 91.9, 93.1, 94.3, 95.5, 96.7, 97.3,
     97.3],
]
# Morning hour lookup tables for high (H) RH class
H = [
    [9999.0, 17.5, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 72.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0,
     83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 100.9,
     101.0],
    [600.0, 28.2, 33.4, 37.9, 42.9, 45.6, 48.5, 51.7, 55.1, 56.5, 58.0, 58.8, 59.5, 60.3, 61.2, 62.0, 62.9, 63.7, 64.6,
     65.5, 66.5, 67.4, 68.4, 69.4, 70.5, 71.6, 72.7, 73.8, 75.0, 76.2, 77.4, 78.7, 80.0, 81.4, 82.7, 84.1, 85.4, 86.3,
     86.3],
    [700.0, 30.0, 34.8, 39.0, 43.8, 46.5, 49.4, 52.5, 55.9, 57.3, 58.8, 59.6, 60.4, 61.2, 62.1, 62.9, 63.8, 64.7, 65.7,
     66.6, 67.6, 68.6, 69.6, 70.7, 71.8, 72.9, 74.1, 75.3, 76.5, 77.8, 79.1, 80.5, 81.9, 83.3, 84.8, 86.2, 87.6, 88.4,
     88.4],
    [800.0, 31.9, 36.2, 40.2, 44.8, 47.4, 50.2, 53.3, 56.7, 58.2, 59.7, 60.5, 61.3, 62.2, 63.0, 63.9, 64.8, 65.7, 66.7,
     67.7, 68.7, 69.8, 70.8, 71.9, 73.1, 74.3, 75.5, 76.8, 78.1, 79.4, 80.8, 82.3, 83.8, 85.3, 86.9, 88.4, 89.8, 90.6,
     90.6],
    [900.0, 37.7, 42.1, 46.1, 50.5, 52.9, 55.5, 58.4, 61.5, 62.8, 64.2, 64.9, 65.6, 66.4, 67.1, 67.9, 68.7, 69.5, 70.4,
     71.3, 72.1, 73.1, 74.0, 75.0, 76.0, 77.0, 78.1, 79.2, 80.3, 81.5, 82.7, 84.0, 85.3, 86.7, 88.1, 89.5, 90.8, 91.7,
     91.7],
    [1000.0, 44.4, 48.9, 52.7, 56.8, 59.1, 61.4, 63.9, 66.7, 67.8, 69.0, 69.6, 70.2, 70.9, 71.5, 72.2, 72.9, 73.6, 74.3,
     75.0, 75.8, 76.6, 77.3, 78.2, 79.0, 79.9, 80.8, 81.7, 82.6, 83.6, 84.7, 85.8, 86.9, 88.0, 89.3, 90.5, 91.8, 92.8,
     92.8],
    [1100.0, 52.1, 56.5, 60.2, 63.9, 65.9, 67.9, 70.1, 72.3, 73.3, 74.3, 74.9, 75.5, 76.1, 76.6, 77.2, 77.8, 78.4, 79.0,
     79.5, 80.1, 80.7, 81.2, 81.8, 82.4, 82.9, 83.5, 84.0, 84.6, 85.1, 85.6, 86.2, 86.9, 88.0, 89.3, 90.5, 91.8, 92.8,
     92.8],
    [1159.0, 60.9, 65.2, 68.6, 71.8, 73.5, 75.1, 76.7, 78.4, 79.1, 79.8, 80.2, 80.5, 80.8, 81.2, 81.5, 81.8, 82.1, 82.5,
     82.8, 83.1, 83.4, 83.7, 84.0, 84.3, 84.6, 84.9, 85.2, 85.5, 85.8, 86.1, 86.4, 86.9, 88.0, 89.3, 90.5, 91.8, 92.8,
     92.8],
    [1200.0, 60.9, 65.2, 68.6, 71.8, 73.5, 75.1, 76.7, 78.4, 79.1, 79.8, 80.2, 80.5, 80.8, 81.2, 81.5, 81.8, 82.1, 82.5,
     82.8, 83.1, 83.4, 83.7, 84.0, 84.3, 84.6, 84.9, 85.2, 85.5, 85.8, 86.1, 86.4, 86.9, 88.0, 89.3, 90.5, 91.8, 92.8,
     92.8],
]
# Main interpolation table for hours beyond 12:00
MAIN = [
    [9999.0, 17.5, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 72.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0,
     83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 100.9,
     101.0],
    [100.0, 23.4, 32.9, 40.5, 47.8, 51.4, 54.9, 58.3, 61.8, 63.3, 64.8, 65.5, 66.3, 67.1, 67.9, 68.8, 69.6, 70.5, 71.4,
     72.3, 73.2, 74.1, 75.1, 76.1, 77.1, 78.1, 79.1, 80.2, 81.3, 82.4, 83.5, 84.7, 85.9, 87.1, 88.3, 89.5, 90.7, 91.6,
     91.6],
    [200.0, 24.3, 33.0, 39.9, 46.8, 50.2, 53.6, 56.9, 60.4, 61.8, 63.4, 64.1, 64.9, 65.7, 66.5, 67.4, 68.2, 69.1, 70.0,
     70.9, 71.8, 72.7, 73.7, 74.7, 75.7, 76.7, 77.8, 78.9, 80.0, 81.1, 82.3, 83.4, 84.7, 85.9, 87.2, 88.4, 89.6, 90.5,
     90.5],
    [300.0, 25.2, 33.1, 39.4, 45.8, 49.0, 52.3, 55.6, 59.0, 60.5, 62.0, 62.7, 63.5, 64.3, 65.1, 66.0, 66.8, 67.7, 68.6,
     69.5, 70.4, 71.4, 72.3, 73.3, 74.4, 75.4, 76.5, 77.6, 78.7, 79.8, 81.0, 82.2, 83.5, 84.7, 86.0, 87.3, 88.5, 89.4,
     89.4],
    [400.0, 26.2, 33.2, 38.9, 44.8, 47.9, 51.0, 54.3, 57.7, 59.1, 60.6, 61.4, 62.2, 63.0, 63.8, 64.6, 65.5, 66.3, 67.2,
     68.1, 69.1, 70.0, 71.0, 72.0, 73.0, 74.1, 75.2, 76.3, 77.4, 78.6, 79.8, 81.0, 82.3, 83.6, 84.9, 86.2, 87.5, 88.4,
     88.4],
    [500.0, 27.2, 33.3, 38.4, 43.9, 46.7, 49.8, 52.9, 56.4, 57.8, 59.3, 60.1, 60.8, 61.6, 62.5, 63.3, 64.2, 65.0, 65.9,
     66.8, 67.8, 68.7, 69.7, 70.7, 71.7, 72.8, 73.9, 75.0, 76.2, 77.4, 78.6, 79.8, 81.1, 82.5, 83.8, 85.2, 86.4, 87.3,
     87.3],
    [559.0, 28.2, 33.4, 37.9, 42.9, 45.7, 48.6, 51.7, 55.1, 56.5, 58.0, 58.8, 59.6, 60.4, 61.2, 62.0, 62.9, 63.8, 64.6,
     65.6, 66.5, 67.5, 68.4, 69.5, 70.5, 71.6, 72.7, 73.8, 75.0, 76.2, 77.4, 78.7, 80.0, 81.4, 82.7, 84.1, 85.4, 86.3,
     86.3],
    [600.0, 28.2, 33.4, 37.9, 42.9, 45.7, 48.6, 51.7, 55.1, 56.5, 58.0, 58.8, 59.6, 60.4, 61.2, 62.0, 62.9, 63.8, 64.6,
     65.6, 66.5, 67.5, 68.4, 69.5, 70.5, 71.6, 72.7, 73.8, 75.0, 76.2, 77.4, 78.7, 80.0, 81.4, 82.7, 84.1, 85.4, 86.3,
     86.3],
    [1200.0, 17.5, 27.7, 34.4, 40.9, 44.5, 48.2, 52.5, 57.3, 59.4, 61.7, 62.9, 64.2, 65.5, 66.9, 68.5, 70.5, 73.8, 76.4,
     78.4, 80.0, 81.5, 82.8, 84.0, 85.2, 86.3, 87.5, 88.6, 89.7, 90.8, 91.9, 92.9, 94.0, 95.0, 96.0, 97.0, 97.9, 98.7,
     98.7],
    [1300.0, 17.5, 28.3, 35.8, 43.2, 47.2, 51.5, 56.0, 61.0, 63.2, 65.5, 66.7, 67.9, 69.3, 70.7, 72.2, 73.9, 76.3, 78.2,
     79.8, 81.1, 82.4, 83.7, 84.8, 86.0, 87.1, 88.2, 89.3, 90.4, 91.4, 92.5, 93.5, 94.6, 95.6, 96.6, 97.6, 98.5, 99.3,
     99.3],
    [1400.0, 17.5, 29.0, 37.2, 45.6, 50.1, 54.8, 59.8, 65.1, 67.3, 69.6, 70.8, 72.0, 73.3, 74.6, 76.1, 77.4, 78.7, 79.9,
     81.1, 82.3, 83.4, 84.6, 85.7, 86.8, 87.9, 88.9, 90.0, 91.0, 92.1, 93.1, 94.1, 95.1, 96.1, 97.1, 98.1, 99.1, 100.0,
     100.0],
    [1500.0, 17.5, 29.5, 38.6, 47.8, 52.5, 57.4, 62.4, 67.5, 69.6, 71.8, 72.9, 74.0, 75.1, 76.3, 77.5, 78.7, 79.9, 81.0,
     82.1, 83.2, 84.2, 85.3, 86.4, 87.4, 88.5, 89.5, 90.5, 91.5, 92.6, 93.6, 94.6, 95.6, 96.6, 97.6, 98.6, 99.6, 100.4,
     100.4],
    [1600.0, 17.5, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 72.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0,
     83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.1, 93.1, 94.1, 95.1, 96.1, 97.1, 98.1, 99.1, 100.1, 101.0,
     101.0],
    [1700.0, 17.8, 30.6, 40.8, 51.0, 56.1, 61.0, 65.8, 70.4, 72.2, 74.0, 75.0, 75.9, 76.8, 77.8, 78.7, 79.7, 80.6, 81.6,
     82.6, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.6, 98.6, 99.6, 100.4,
     100.4],
    [1800.0, 18.0, 31.1, 41.6, 52.0, 57.1, 62.0, 66.6, 70.7, 72.3, 74.0, 74.9, 75.7, 76.6, 77.5, 78.4, 79.3, 80.2, 81.2,
     82.1, 83.0, 84.0, 84.9, 85.9, 86.9, 87.9, 88.9, 89.9, 90.9, 91.9, 92.9, 93.9, 95.0, 96.0, 97.1, 98.1, 99.1, 99.9,
     99.9],
    [1900.0, 18.5, 31.8, 42.4, 52.6, 57.5, 62.0, 66.2, 70.0, 71.6, 73.2, 74.0, 74.8, 75.7, 76.5, 77.4, 78.2, 79.1, 80.0,
     80.9, 81.8, 82.8, 83.7, 84.6, 85.6, 86.6, 87.5, 88.5, 89.5, 90.5, 91.5, 92.6, 93.6, 94.6, 95.7, 96.7, 97.8, 98.6,
     98.6],
    [2000.0, 19.1, 32.5, 43.2, 53.3, 57.9, 62.0, 65.9, 69.4, 70.9, 72.4, 73.1, 73.9, 74.7, 75.5, 76.3, 77.2, 78.0, 78.9,
     79.8, 80.6, 81.5, 82.5, 83.4, 84.3, 85.3, 86.2, 87.2, 88.2, 89.2, 90.2, 91.2, 92.3, 93.3, 94.3, 95.4, 96.4, 97.4,
     97.4],
    [2100.0, 19.9, 32.5, 42.6, 52.1, 56.5, 60.5, 64.3, 67.8, 69.3, 70.8, 71.5, 72.3, 73.1, 73.9, 74.8, 75.6, 76.5, 77.3,
     78.2, 79.1, 80.0, 80.9, 81.9, 82.8, 83.8, 84.8, 85.8, 86.8, 87.8, 88.8, 89.9, 90.9, 92.0, 93.1, 94.2, 95.2, 96.2,
     96.2],
    [2200.0, 20.7, 32.6, 42.1, 51.0, 55.2, 59.1, 62.7, 66.2, 67.7, 69.2, 70.0, 70.8, 71.6, 72.4, 73.2, 74.1, 74.9, 75.8,
     76.7, 77.6, 78.5, 79.4, 80.4, 81.3, 82.3, 83.3, 84.3, 85.4, 86.4, 87.5, 88.6, 89.6, 90.8, 91.9, 93.0, 94.1, 95.0,
     95.0],
    [2300.0, 21.6, 32.7, 41.5, 50.0, 53.9, 57.6, 61.2, 64.7, 66.2, 67.7, 68.5, 69.3, 70.1, 70.9, 71.7, 72.5, 73.4, 74.3,
     75.2, 76.1, 77.0, 77.9, 78.9, 79.9, 80.9, 81.9, 82.9, 84.0, 85.0, 86.1, 87.2, 88.4, 89.5, 90.7, 91.8, 92.9, 93.9,
     93.9],
    [2400.0, 22.5, 32.8, 41.0, 48.9, 52.7, 56.3, 59.8, 63.3, 64.7, 66.2, 67.0, 67.8, 68.6, 69.4, 70.2, 71.1, 71.9, 72.8,
     73.7, 74.6, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.6, 83.7, 84.8, 86.0, 87.1, 88.3, 89.5, 90.7, 91.8, 92.7,
     92.7],
    [2500.0, 23.4, 32.9, 40.5, 47.8, 51.4, 54.9, 58.3, 61.8, 63.3, 64.8, 65.5, 66.3, 67.1, 67.9, 68.8, 69.6, 70.5, 71.4,
     72.3, 73.2, 74.1, 75.1, 76.1, 77.1, 78.1, 79.1, 80.2, 81.3, 82.4, 83.5, 84.7, 85.9, 87.1, 88.3, 89.5, 90.7, 91.6,
     91.6],
]
# RH classification structure
RHCLASS = [
    [600, 630],
    [700, 730],
    [800, 830],
    [900, 930],
    [1000, 1030],
    [1100, 1130],
    [1159, 1200],
    [1200, 1200],
    [87, 3],
    [77, 3],
    [67, 3],
    [62, 3],
    [57, 3],
    [54.5, 3],
    [52, 3],
    [52, 3],
    [87, 2],
    [77, 2],
    [67, 2],
    [62, 2],
    [57, 2],
    [54.5, 2],
    [52, 2],
    [52, 2],
    [68, 1],
    [58, 1],
    [48, 1],
    [43, 1],
    [38, 1],
    [35.5, 1],
    [33, 1],
    [33, 1],
]


# THIS IS THE DIRECT PYTHON INTERPRETATION FROM THE C++ CODE IN WISE_FWI_MODULE.CPP
# # Time helper class to mimic WTimeSpan
# class TimeSpan:
#     """
#     Represents a time interval using hours and minutes.
#
#     Provides methods to access hours, minutes, and compute the total number of minutes.
#     """
#
#     def __init__(self, hours: int, minutes: int):
#         self.hours = hours
#         self.minutes = minutes
#
#     def get_hours(self) -> int:
#         """Return the hour component."""
#         return self.hours
#
#     def get_minutes(self) -> int:
#         """Return the minute component."""
#         return self.minutes
#
#     def get_total_minutes(self) -> int:
#         """Return total minutes represented by this timespan."""
#         return self.hours * 60 + self.minutes
#
#
# def interpolate(i1: float, i2: float, i3: float, i4: float, fraction: float, ts: TimeSpan) -> float:
#     """
#     Perform bilinear interpolation between four points in a grid based on a fractional x/y position.
#
#     :param i1: Lower-left value
#     :param i2: Lower-right value
#     :param i3: Upper-left value
#     :param i4: Upper-right value
#     :param fraction: X interpolation fraction between columns
#     :param ts: A TimeSpan object providing the current time (used for vertical interpolation)
#     :return: Interpolated FFMC value
#     """
#     i12 = i1 + ((i2 - i1) * fraction)  # Interpolate across first row
#     i34 = i3 + ((i4 - i3) * fraction)  # Interpolate across second row
#
#     # Interpolate between i12 and i34 vertically
#     divisor = 59.0 if ts.get_hours() == 11 else 60.0
#     return i12 + ((i34 - i12) / divisor) * ts.get_minutes()
#
#
# def interpolate_table(table: list[list[float]], ffmc: float, tindex: int, ts: TimeSpan) -> float:
#     """
#     Interpolate FFMC value from a 2D table (L, M, H, or MAIN) based on current FFMC and time.
#
#     :param table: A 2D lookup table for FFMC (e.g., L, M, H)
#     :param ffmc: Current FFMC value (must be between 17.5 and 101)
#     :param tindex: Row index based on time-of-day
#     :param ts: TimeSpan object for interpolation reference
#     :return: Interpolated FFMC value
#     """
#     i = 1
#     while ffmc >= table[0][i]:
#         i += 1
#     i -= 1
#
#     fraction = (ffmc - table[0][i]) / (table[0][i + 1] - table[0][i])
#
#     return interpolate(table[tindex][i], table[tindex][i + 1],
#                        table[tindex + 1][i], table[tindex + 1][i + 1],
#                        fraction, ts)
#
#
# def hourlyFFMC_lawson_calc(ffmc: float, ts: TimeSpan, rh: float) -> float:
#     """
#     Calculate the hourly-adjusted FFMC based on the Lawson interpolation method.
#
#     :param ffmc: Initial FFMC value (typically from daily FWI system)
#     :param ts: TimeSpan object representing local solar time (hours, minutes)
#     :param rh: Relative Humidity (0–100%)
#     :return: Adjusted hourly FFMC value, or -98.0 if input FFMC is invalid
#     """
#     if ffmc < 0.0 or ffmc > 101.0:
#         return -98.0  # Invalid input
#
#     ffmc = max(ffmc, 17.5)
#     rh = max(0.0, min(100.0, rh))
#     rh = round(rh)
#     if rh < 1.0:
#         rh = 95
#
#     hour = ts.get_hours()
#     minutes = ts.get_minutes()
#
#     # Morning transition period (06:00 to 12:00)
#     if 6 <= hour <= 11:
#         tindex = 0
#         for i in range(8):
#             if 100 * hour < RHCLASS[0][i][0]:
#                 tindex = i
#                 break
#
#         # Determine RH class: Low (L), Medium (M), High (H)
#         if minutes <= 30:
#             if rh > RHCLASS[1][tindex - 1][0]:
#                 rh_class = 'H'
#             elif rh < RHCLASS[3][tindex - 1][0]:
#                 rh_class = 'L'
#             else:
#                 rh_class = 'M'
#         else:
#             if rh > RHCLASS[1][tindex][0]:
#                 rh_class = 'H'
#             elif rh < RHCLASS[3][tindex][0]:
#                 rh_class = 'L'
#             else:
#                 rh_class = 'M'
#
#         # Use appropriate table for interpolation
#         if rh_class == 'L':
#             return interpolate_table(L, ffmc, tindex, ts)
#         elif rh_class == 'M':
#             return interpolate_table(M, ffmc, tindex, ts)
#         else:
#             return interpolate_table(H, ffmc, tindex, ts)
#
#     # Afternoon/evening period: use MAIN table
#     hour_val = hour * 100 + minutes
#     if hour_val < 100:
#         hour_val += 2400  # Adjust for times past midnight
#
#     tindex = 1
#     while hour_val >= MAIN[tindex][0]:
#         tindex += 1
#     tindex -= 1
#
#     i = 1
#     while ffmc >= MAIN[0][i]:
#         i += 1
#     i -= 1
#
#     fraction = (ffmc - MAIN[0][i]) / (MAIN[0][i + 1] - MAIN[0][i])
#     return interpolate(MAIN[tindex][i], MAIN[tindex][i + 1],
#                        MAIN[tindex + 1][i], MAIN[tindex + 1][i + 1],
#                        fraction, ts)
#

# def hourlyFFMC_lawson(ffmc: Union[float, np.ndarray],
#                       rh: Union[float, np.ndarray],
#                       hour: int,
#                       minute: int) -> Union[float, np.ndarray]:
#     """
#     Vectorized version of hourlyFFMC_lawson that accepts FFMC and RH as arrays or scalars.
#     Hour and minute must be scalars (applied uniformly to all inputs).
#
#     :param ffmc: Initial FFMC values
#     :param rh: Relative humidity (%) values
#     :param hour: Hour of day
#     :param minute: Minute
#     :return: Predicted hourly FFMC value(s)
#     """
#     ffmc = np.atleast_1d(ffmc)
#     rh = np.atleast_1d(rh)
#
#     if ffmc.shape != rh.shape:
#         raise ValueError('ffmc and rh must have the same shape')
#
#     ts = TimeSpan(hour, minute)
#     result = np.full(ffmc.shape, -98.0, dtype=np.float32)
#
#     # Currently processes each element independently due to the table interpolations
#     for idx in np.ndindex(ffmc.shape):
#         result[idx] = hourlyFFMC_lawson_calc(float(ffmc[idx]), ts, float(rh[idx]))
#
#     return result if result.size > 1 else result[0]


def hourly_ffmc_lawson_vectorized(
        ffmc: Union[float, np.ndarray],
        rh: Union[float, np.ndarray],
        hour: int,
        minute: int
) -> Union[float, np.ndarray]:
    """
    Vectorized implementation of the Lawson hourly FFMC interpolation.

    :param ffmc: Array of initial FFMC values (float32)
    :param hour: Array of hour values (int)
    :param minute: Array of minute values (int)
    :param rh: Array of relative humidity values (%)
    :return: Array of hourly FFMC values
    """
    main_tbl = np.asarray(MAIN, dtype=np.float64)
    low_tbl = np.asarray(L, dtype=np.float64)
    med_tbl = np.asarray(M, dtype=np.float64)
    high_tbl = np.asarray(H, dtype=np.float64)
    rhclass_tbl = np.asarray(RHCLASS, dtype=np.float64)

    ffmc_ma = np.ma.asarray(ffmc, dtype=np.float64)
    rh_ma = np.ma.asarray(rh, dtype=np.float64)
    hour_arr = np.asarray(hour, dtype=np.int32)
    minute_arr = np.asarray(minute, dtype=np.int32)

    ffmc_data, rh_data, hour_data, minute_data = np.broadcast_arrays(
        np.ma.getdata(ffmc_ma),
        np.ma.getdata(rh_ma),
        hour_arr,
        minute_arr
    )
    ffmc_mask, rh_mask = np.broadcast_arrays(
        np.ma.getmaskarray(ffmc_ma),
        np.ma.getmaskarray(rh_ma)
    )
    combined_mask = ffmc_mask | rh_mask

    ffmc_data = np.clip(ffmc_data, 17.5, 101.0)
    rh_data = np.clip(np.rint(rh_data), 1, 100).astype(np.int32)

    ffmc_out = np.full(ffmc_data.shape, np.nan, dtype=np.float64)

    hour_val = hour_data * 100 + minute_data
    hour_val = np.where(hour_val < 100, hour_val + 2400, hour_val)

    is_morning = (~combined_mask) & (hour_data >= 6) & (hour_data <= 11)
    is_main = (~combined_mask) & (~((hour_data >= 6) & (hour_data <= 11)))

    if np.any(is_main):
        ffmc_main = ffmc_data[is_main]
        hour_main = hour_data[is_main]
        minute_main = minute_data[is_main]
        hour_val_main = hour_val[is_main]

        tindex = np.searchsorted(main_tbl[:, 0], hour_val_main, side='right') - 1
        tindex = np.clip(tindex, 1, main_tbl.shape[0] - 2)

        fidx = np.searchsorted(main_tbl[0], ffmc_main, side='right') - 1
        fidx = np.clip(fidx, 1, main_tbl.shape[1] - 2)

        frac = (ffmc_main - main_tbl[0, fidx]) / (main_tbl[0, fidx + 1] - main_tbl[0, fidx])
        i12 = main_tbl[tindex, fidx] + (main_tbl[tindex, fidx + 1] - main_tbl[tindex, fidx]) * frac
        i34 = main_tbl[tindex + 1, fidx] + (main_tbl[tindex + 1, fidx + 1] - main_tbl[tindex + 1, fidx]) * frac

        divisor = np.where(hour_main == 11, 59.0, 60.0)
        ffmc_out[is_main] = i12 + (i34 - i12) * (minute_main / divisor)

    if np.any(is_morning):
        rh_cutoff = rhclass_tbl[:8, 0]
        rh_class_l = rhclass_tbl[24:32, 0]
        rh_class_h = rhclass_tbl[8:16, 0]

        ffmc_morning = ffmc_data[is_morning]
        rh_morning = rh_data[is_morning]
        hour_morning = hour_data[is_morning]
        minute_morning = minute_data[is_morning]
        hour_val_morning = hour_morning * 100 + minute_morning

        tindex = np.searchsorted(rh_cutoff, hour_val_morning, side='right')
        tindex = np.clip(tindex, 1, 7)

        rh_class = np.full(rh_morning.shape, 'M', dtype='<U1')
        rh_class[rh_morning > rh_class_h[tindex]] = 'H'
        rh_class[rh_morning < rh_class_l[tindex]] = 'L'

        table_map = {'L': low_tbl, 'M': med_tbl, 'H': high_tbl}
        out_vals = np.zeros(ffmc_morning.shape, dtype=np.float64)

        for cls in ['L', 'M', 'H']:
            sel_idx = np.flatnonzero(rh_class == cls)
            if sel_idx.size == 0:
                continue

            ffmc_sel = ffmc_morning[sel_idx]
            minute_sel = minute_morning[sel_idx]
            hour_sel = hour_morning[sel_idx]
            t_sel = tindex[sel_idx]
            tbl = table_map[cls]

            for local_i, (f, m, h, ti) in enumerate(zip(ffmc_sel, minute_sel, hour_sel, t_sel)):
                iidx = np.searchsorted(tbl[0], f, side='right') - 1
                iidx = int(np.clip(iidx, 1, tbl.shape[1] - 2))
                frac = (f - tbl[0, iidx]) / (tbl[0, iidx + 1] - tbl[0, iidx])
                i12 = tbl[ti, iidx] + (tbl[ti, iidx + 1] - tbl[ti, iidx]) * frac
                i34 = tbl[ti + 1, iidx] + (tbl[ti + 1, iidx + 1] - tbl[ti + 1, iidx]) * frac
                div = 59.0 if h == 11 else 60.0
                out_vals[sel_idx[local_i]] = i12 + (i34 - i12) * (m / div)

        ffmc_out[is_morning] = out_vals

    if ffmc_out.shape == ():
        return float(ffmc_out)
    return ffmc_out
