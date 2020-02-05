import sys, os
import numpy as np
import pandas as pd
import pickle



def color_palette():
    """
    Defines the color palettes used in the thesis.
    """
    c_blue_med = "#053B89"
    c_blue_dark = "#072159"
    c_blue_light = "#1261A0"
    c_red = "#BE0209"
    c_red_light = "#e2514a"
    c_orange = "#E96F36"
    c_orange_light = "#fca55d"
    c_gray = "#282739"

    colors = {"blue_dark": "#072159",
              "blue_med": "#053B89",
              "blue_light": "#1261A0",
              "red": "#BE0209",
              "red_light": "#e2514a",
              "orange": "#E96F36",
              "orange_light": "#fca55d",
              "gray": "#282739"}
    palette = [c_blue_dark, c_blue_med, c_blue_light, c_red, c_red_light, c_orange, c_orange_light, c_gray]
    return colors, palette