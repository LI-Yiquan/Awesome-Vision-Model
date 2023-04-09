import math
import cv2
import numpy as np

class Wall:

    def __init__(self,size=2,res=64,block_size=32,
                 uv_size=64,foresshortening=False,
                 depth=1):
