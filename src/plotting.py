#!/usr/bin/env python3

import imageio
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pykep as pk
import os

import constants
import utils

def plot_planets(asteroids_data: pl.DataFrame,
                 axes: plt.axes,
                 t0: float = pk.epoch_from_iso_string(constants.ISO_T_START),
                 tf: float = pk.epoch_from_iso_string(constants.ISO_T_END),
                 s: int = 12,
                 marker: str = "o",
                 color: str = None):
    '''
    Function to plot asteroids within
    polars DataFrame

    Parameters:
        - asteroids_data (pl.DataFrame): Asteroids data to be plotted
        - axes (plt.axes): Axes to plot asteroids into the same figure.
        - t0 (float): Start time to plot planet
        - tf (float): End time to plot planet
        - s (int): Planet size (Default 12)

    Returns:
        - plt.axes: Plot axes
    '''

    # Define a lookup table to select a color in based of material type
    color_lookup = {0: "gold",
                    1: "red",
                    2: "green",
                    3: "blue"}

    # Iterate over ID field
    ids = list(asteroids_data["ID"])
    for idx in ids:
        # Extract planet dataframe
        planet_data = asteroids_data.filter(pl.col("ID") == idx)

        # Convert it to planet object
        planet = utils.convert_to_planet(planet_data)

        # Select color according asteroid material type
        if not color:
            _color = color_lookup[planet_data["Material Type"].item()]
        else:
            _color = color

        # Use of pykep plotting module
        axes = pk.orbit_plots.plot_planet(planet,
                                          axes=axes,
                                          s=s,
                                          marker=marker,
                                          t0=t0,
                                          tf=tf,
                                          alpha=0, # Don't show orbit plot
                                          color=_color)

        # Set axis limits
        axes.set_xlim([-3e10, 3e10])
        axes.set_ylim([-3e10, 3e10])
        axes.set_zlim([-8e9, 8e9])

    return axes

def animate_planets(asteroid_data: pl.DataFrame,
                    start_time: float = 372242.0,
                    time: float = constants.TIME_OF_MISSION,
                    filename: str = "planets.gif"):
    '''
    Generate an animated GIF of asteroid orbits within [start_time, time]

    Parameters:
        - asteroids_data (pl.DataFrame): Asteroids data to be plotted
        - start_time (float): Reference start time (Default: T_START)
        - time (float): End of animated plot (Default: TIME_OF_MISSION)

    Returns:
        - None
    '''

    images = []
    # Iterate over time ceiling end of animated reference
    for t in range(1, int(np.ceil(time))):

        # Generate a new axis for current plot
        axes = plt.figure().add_subplot(projection="3d")

        # Plot asteroid position at t
        axes = plot_planets(asteroids_data=asteroid_data,
                            axes=axes,
                            t0=start_time + t)

        # Display t into the title of current plot
        axes.set_title(f"t = {t}")

        # Save plot as fig_XXXX.png file
        time_string = str(t).zfill(4)
        temp_filename = f"fig_{time_string}.png"
        temp_filename = temp_filename.zfill(4)
        plt.savefig(temp_filename)
        plt.close()

        # Open file with imageio imread to generate the animation
        images.append(imageio.imread(temp_filename))

        # Remove plot file
        os.remove(temp_filename)

    # Create GIF file
    imageio.mimsave(filename, images)


def animate_journey(asteroids_data: pl.DataFrame,
                    asteroids: list,
                    time_of_arrival: list,
                    start_time: float = 372242.0,
                    filename: str = "animate_journey.gif"):

    images = []
    end_time = int(np.ceil(time_of_arrival[-1]))
    for t in range(1, end_time):

        # Extract the closest value between current t and time of arrival
        # to extract the current asteroid where the rover is.
        min_value = min(time_of_arrival, key= lambda x:abs(x - t))
        min_index = time_of_arrival.index(min_value)

        current_asteroid_id = asteroids[min_index]
        current_asteroid_data = asteroids_data.filter(
            pl.col("ID") == current_asteroid_id
        )
        others_asteroid_data = asteroids_data.filter(
            pl.col("ID") != current_asteroid_id
        )

        # Generate a new axis for current plot
        axes = plt.figure().add_subplot(projection="3d")

        # Plot asteroids position at t
        axes = plot_planets(asteroids_data=current_asteroid_data,
                            axes=axes,
                            s=30,
                            marker="*",
                            t0=start_time + t)
        axes = plot_planets(asteroids_data=others_asteroid_data,
                            axes=axes,
                            s=5,
                            t0=start_time + t)

        # Display t into the title of current plot
        axes.set_title(f"t = {t}")

        # Save plot as fig_XXXX.png file
        time_string = str(t).zfill(4)
        temp_filename = f"fig_{time_string}.png"
        temp_filename = temp_filename.zfill(4)
        plt.savefig(temp_filename)
        plt.close()

        # Open file with imageio imread to generate the animation
        images.append(imageio.imread(temp_filename))

        # Remove plot file
        os.remove(temp_filename)

    # Create GIF file
    imageio.mimsave(filename, images)

