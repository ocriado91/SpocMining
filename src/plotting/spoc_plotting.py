# -*- coding: utf-8 -*-


"""
Plotting module

Functions contained:
    - plotting_asteroids_by_material
    - plotting_zenith_by_material
    - plot_asteroids
    - plot_zenith_asteroids
    - plot_asteroids_by_material
    - plot_zenith_asteroids_by_material
    - animate_orbit
    - plot_distance
    - plot_deltaV
    - plot_mass_distribution
"""
#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import glob
from locale import normalize
from statistics import mode
import imageio
import logging
import logging
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pykep as pk

# Own Libs
from asteroid import Asteroid
import constants


#######################################################################

# Set logging level and format
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T_START = pk.epoch_from_iso_string(constants.ISO_T_START)
T_END = pk.epoch_from_iso_string(constants.ISO_T_END)

def plot_mass_distribution(asteroidList: list,
                           figureFolder: str,
                           figurename: str = 'mass_histogram.png'):

    """
    Plot mass distribution of asteroids in list

    Parameters
    ----------
    asteroidList : list
        List of asteroids
    figureFolder: str
        Path to store plots
    figurename : str
        Name of figure

    Returns
    -------
    None

    """

    logger.info('Generating mass distribution plotting')
    goldMass = 0
    platinumMass = 0
    nickelMass = 0
    propellantMass = 0
    for asteroid in asteroidList:
        if asteroid.materialType == 'Gold':
            goldMass += asteroid.normalizedMass
        elif asteroid.materialType == 'Platinum':
            platinumMass += asteroid.normalizedMass
        elif asteroid.materialType == 'Nickel':
            nickelMass += asteroid.normalizedMass
        elif asteroid.materialType == 'Propellant':
            propellantMass += asteroid.normalizedMass

    massList = [goldMass, platinumMass, nickelMass, propellantMass]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot bar with different colors
    ax.bar(['Gold', 'Platinum', 'Nickel', 'Propellant'],
            massList,
            color=['r', 'g', 'b', 'c'])

    # Set plot properties
    ax.set_title('Mass distribution')
    ax.set_ylabel('Mass (kg)')
    ax.set_xlabel('Material type')
    figurename = os.path.join(figureFolder, figurename)
    fig.savefig(figurename)


def plotting_asteroids_by_material(asteroidList: list,
                               materialType: str,
                               ax: plt.Axes = None,
                               color: str = 'k',
                               alpha: float = 0.5,
                               size: int = 5):
    """
    Plot asteroids by material type

    Parameters
    ----------
    asteroidList : list
        List of asteroids

    materialType : str
        Material type to plot

    ax : plt.Axes
        Axes to plot on

    color : str
        Color to plot

    alpha : float
        Alpha value to plot

    size : int
        Size to plot

    Returns
    -------
    None

    """

    asteroidsByMaterial = [asteroid for asteroid in asteroidList \
        if asteroid.materialType == materialType]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    xCoord = []
    yCoord = []
    zCoord = []
    for asteroid in asteroidsByMaterial:
        x, y, z = asteroid.get_coordinates()
        xCoord.append(x)
        yCoord.append(y)
        zCoord.append(z)

    ax.scatter(xCoord, yCoord, zCoord,
               color=color,
               alpha=alpha,
               label=materialType,
               s=size)

def plotting_zenith_by_material(asteroidList: list,
                                materialType: str,
                                ax: plt.Axes = None,
                                color: str = 'k',
                                alpha: float = 0.5,
                                size: int = 5):
    """
    Zenith asteroids by material type

    Parameters
    ----------
    asteroidList : list
        List of asteroids

    materialType : str
        Material type to plot

    ax : plt.Axes
        Axes to plot on

    color : str
        Color to plot

    alpha : float
        Alpha value to plot

    size : int
        Size to plot

    Returns
    -------
    None

    """

    asteroidsByMaterial = [asteroid for asteroid in asteroidList \
        if asteroid.materialType == materialType]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    xCoord = []
    yCoord = []
    for asteroid in asteroidsByMaterial:
        x, y, _ = asteroid.get_coordinates()
        xCoord.append(x)
        yCoord.append(y)

    ax.scatter(xCoord, yCoord,
               color=color,
               alpha=alpha,
               label=materialType,
               s=size)


def plot_asteroids(asteroidList: list,
                   figureFolder: str,
                   figurename: str = 'asteroids.png'):
    """
    Plot all asteroids in list at the same plot

    Parameters
    ----------
    asteroidList : list
        List of asteroids

    figurename : str
        Name of the figure to save

    Returns
    -------
    None

    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotting_asteroids_by_material(asteroidList,
                               'Gold',
                               ax,
                               color='r')
    plotting_asteroids_by_material(asteroidList,
                               'Platinum',
                               ax,
                               color='g')
    plotting_asteroids_by_material(asteroidList,
                               'Nickel',
                               ax,
                               color='b')
    plotting_asteroids_by_material(asteroidList,
                               'Propellant',
                               ax,
                               color='c')

    handles, labels = plt.gca().get_legend_handles_labels()
    byLabel = dict(zip(labels, handles))

    # Set title
    ax.title.set_text("Asteroids")

    # Set legend box below the plot
    ax.legend(byLabel.values(), byLabel.keys(),
                loc='lower left',
                bbox_to_anchor=(0, -0.1, 1, -0.1),
                ncol=5,
                mode="expand",
                borderaxespad=0.,
                )

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save figure
    figurename = os.path.join(figureFolder, figurename)
    plt.savefig(figurename)

def plot_zenith_asteroids(asteroidList: list,
                          figureFolder: str,
                          figurename: str = 'zenith_asteroids.png'):
    """
    Plot zenith view of asteroids

    Parameters
    ----------
    asteroidList : list
        List of asteroids

    figurename : str
        Name of the figure to save

    Returns
    -------
    None

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot zenith view by material type
    plotting_zenith_by_material(asteroidList,
                                'Gold',
                                ax,
                                color='r')
    plotting_zenith_by_material(asteroidList,
                                'Platinum',
                                ax,
                                color='g')
    plotting_zenith_by_material(asteroidList,
                                'Nickel',
                                ax,
                                color='b')
    plotting_zenith_by_material(asteroidList,
                                'Propellant',
                                ax,
                                color='c')

    # Avoid duplicated legend
    handles, labels = plt.gca().get_legend_handles_labels()
    byLabel = dict(zip(labels, handles))

    # Set aspect ratio to 1:1
    ax.set_aspect('equal')

    # Set grid
    ax.grid(True)
    ax.grid(which='minor', alpha=0.2)

    # Set title
    ax.title.set_text("Zenith view of asteroids")

    # Set legend box below the plot
    ax.legend(byLabel.values(), byLabel.keys(),
                loc='lower left',
                bbox_to_anchor=(0, -0.1, 1, -0.1),
                ncol=5,
                mode="expand",
                borderaxespad=0., )

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Save figure
    figurename = os.path.join(figureFolder, figurename)
    plt.savefig(figurename)

def plot_asteroids_by_material(asteroidList: list,
                               materialType: str,
                               figurename: str,
                               figureFolder: str,
                               color: str = 'k')->None:

    """
    Plot asteroids by material type

    Parameters
    ----------
    asteroidList : list
        List of asteroids

    materialType : str
        Material type to plot

    color : str
        Color to plot

    figurename : str
        Name of the figure to save

    figureFolder: str
        Folder to store plots

    Returns
    -------
    None

    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits
    ax.set_xlim(-5e10, 5e10)
    ax.set_ylim(-5e10, 5e10)
    ax.set_zlim(-6e9, 6e9)

    # Set title
    ax.title.set_text("Material: {}".format(materialType))
    plotting_asteroids_by_material(asteroidList,
                                   materialType=materialType,
                                   ax=ax,
                                   color=color)
    figurename = os.path.join(figureFolder, figurename)
    plt.savefig(figurename)


def plot_zenith_asteroids_by_material(asteroidList: list,
                                      materialType: str,
                                      figurename: str,
                                      figureFolder: str,
                                      color: str = 'k')->None:

    """
    Plot asteroids by material type

    Parameters
    ----------
    asteroidList : list
        List of asteroids

    materialType : str
        Material type to plot

    color : str
        Color to plot

    figurename : str
        Name of the figure to save

    figureFolder: str
        Folder to store plots

    Returns
    -------
    None

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Set aspect ratio to 1:1
    ax.set_aspect('equal')

    # Set axis limits
    ax.set_xlim(-5e10, 5e10)
    ax.set_ylim(-5e10, 5e10)

    # Set title
    ax.title.set_text("Zenith - material: {}".format(materialType))

    # Plot zenith view by material type
    plotting_zenith_by_material(asteroidList,
                                materialType=materialType,
                                ax=ax,
                                color=color)
    figurename = os.path.join(figureFolder, figurename)
    plt.savefig(figurename)

def plot_orbit(asteroidList: list,
               t0: float = 0.0,
               ax: plt.Axes = None,
               basename: str = 'planets.png'):
    """
    Plot all planets in list at the same plot

    Parameters
    ----------
    asteroidList : list
        List of planets

    figurename : str
        Name of the figure to save

    Returns
    -------
    None

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot first asteroid
    asteroid = asteroidList[0]
    planetObj = asteroid.planetObject
    color = 'k'
    pk.orbit_plots.plot_planet(planetObj,
                                axes=ax,
                                color=color,
                                t0=t0)
    for asteroid in asteroidList[1:]:
        planetObj = asteroid.planetObject
        color = asteroid.materialColor
        pk.orbit_plots.plot_planet(planetObj,
                                axes=ax,
                                color=color,
                                t0=t0)

    # Change point of view
    # ax.view_init(elev=90, azim=0)
    figurename = basename + '_' + str(int(t0/0.1)) + '.png'
    plt.savefig(figurename)

def animate_orbit(asteroidList: list,
                  figureFolder: str,
                  step: float = 0.1,
                  t0 : float = 0.0,
                  tf: float = 10.0,
                  ax: plt.Axes = None,
                  elevation: float = 0.0,
                  azimuth: float = 30.0,
                  figurename: str = 'planets.gif'):
    """
    Plot all planets in list at the same plot

    Parameters
    ----------
    asteroidList : list
        List of planets

    step : float
        Time step between two frames

    t0 : float
        Initial time

    tf : float
        Final time

    figurename : str
        Name of the figure to save

    Returns
    -------
    None

    """

    logger.info('Generating animated plot (.GIF) of asteroids ({})'.format(
        len(asteroidList)))
    # Plot first asteroid
    firstAsteroid = asteroidList[0]
    firstPlanetObj = firstAsteroid.planetObject
    color = 'k'
    for t in np.arange(t0, tf, step):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pk.orbit_plots.plot_planet(firstPlanetObj,
            axes=ax,
            color=color,
            t0=float(t),
            legend=['Asteroid {}'.format(
                firstAsteroid.asteroidId),
            None])
        # Create colormap
        cmap = plt.get_cmap('tab20')
        for i, asteroid in enumerate(asteroidList[1:]):
            planetObj = asteroid.planetObject

            # Plot planet and use color from colormap
            pk.orbit_plots.plot_planet(planetObj,
                axes=ax,
                color=cmap(i),
                t0=float(t),
                legend=['Asteroid {}'.format(
                        asteroid.asteroidId),
                    None])

        # Set title by index
        ax.title.set_text("Orbits at {} s".format(t))

        # Set legend in 3 columns
        ax.legend(loc='center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=3,
                  fontsize=9)


        # Change point of view
        ax.view_init(elev=elevation,
                     azim=azimuth)

        imageName = 'tmp/planets_' + str(int(t)).zfill(4) + '.png'
        logger.debug('Generating {}'.format(imageName))
        plt.tight_layout()
        plt.savefig(imageName)
        plt.close(fig)

    # Create animation using sort files from tmp/planets_*.png
    images = []
    for filename in sorted(glob.glob('tmp/planets_*.png')):
        images.append(imageio.imread(filename))
        # os.remove(filename)

    figurename = os.path.join(figureFolder, figurename)
    imageio.mimsave(figurename, images, duration=0.1)

def plot_distance(asteroids: list,
                  step: float = 1,
                  t0: float = 0.0,
                  tf: float = 30.0,
                  figurename: str = 'distance.png')->None:

    """
    Compute distance between two asteroids

    Parameters
    ----------
    asteroids : list
        List of asteroids
    step : float
        Time step
    t0 : float
        Initial time
    tf : float
        Final time
    figurename : str

    Returns
    -------
    None

    """

    asteroidDistance = dict()
    for asteroid in asteroids[1:]:
        distances = []
        for t in np.arange(t0, tf, step):
            # Coordinates of the first asteroid
            r1, v1 = asteroids[0].planetObject.eph(t)

            x1, y1, z1 = r1

            # Coordinates of the second asteroid
            r2, v2 = asteroid.planetObject.eph(t)

            x2, y2, z2 = r2

            # Distance between the two asteroids
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            distances.append(distance)
            asteroidDistance[asteroid.asteroidId] = distances

    # Plot distance between asteroids
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Discard first value
    for key, value in asteroidDistance.items():
        ax.plot(np.arange(t0, tf, step), value,
            label=f'Asteroid {key}')
    ax.legend()
    plt.savefig(figurename)


def plot_deltaV(asteroids: list,
                step: float = 1,
                tof0: float = 5.0,
                tof1: float = 30.0,
                figurename: str = 'deltaV.png')->None:

    """
    Compute distance between two asteroids

    Parameters
    ----------
    asteroids : list
        List of asteroids
    step : float
        Time step
    t0 : float
        Initial time
    tf : float
        Final time
    figurename : str

    Returns
    -------
    None

    """

    asteroidDV = dict()
    for asteroid in asteroids[1:]:
        deltaV = []
        for tof in np.arange(tof0, tof1, step):
            # Coordinates of the first asteroid
            t1 = T_START.mjd2000 + asteroids[0].normalizedMass *\
                constants.TIME_TO_MINE_FULLY
            logger.info("Time mining {}".format(asteroids[0].normalizedMass *\
                constants.TIME_TO_MINE_FULLY))
            r1, v1 = asteroids[0].planetObject.eph(t1)

            # Coordinates of the second asteroid
            t2 = t1 + tof
            r2, v2 = asteroid.planetObject.eph(t2)

            # Compute Lambert solution
            l = pk.lambert_problem(
                r1=r1,
                r2=r2,
                tof=tof * pk.DAY2SEC,
                mu=constants.MU_TRAPPIST,
                cw=False,
                max_revs=0
            )

            # Compute the delta-v necessary to go there
            # and match its velocity
            DV1 = [a - b for a, b in zip(v1, l.get_v1()[0])]
            DV2 = [a - b for a, b in zip(v2, l.get_v2()[0])]
            deltaV.append(np.linalg.norm(DV1) + np.linalg.norm(DV2))
            asteroidDV[asteroid.asteroidId] = deltaV

    # Plot distance between asteroids
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Discard first value
    for key, value in asteroidDV.items():
        ax.plot(np.arange(tof0, tof1, step), value,
            label=f'Asteroid {key}')
        logger.info("Min DV = %f", min(value))
    ax.set_xlim(0, tof1)
    ax.legend()
    plt.savefig(figurename)

def transfer_window(asteroids: list,
                    step: float = 1,
                    t0: float = 5.0,
                    tf: float = 30.0,
                    tof0: float = 0.0,
                    tof1: float = 50.0,
                    basename: str = 'transfer_window_')->None:

    """
    Plot transfer window into heatmap by pair of asteroids

    Parameters
    ----------
    asteroids : list
        List of asteroids
    step : float
        Time step
    t0 : float
        Initial time
    tf : float
        Final time
    basename : str
        Basename of figurename

    Returns
    -------
    None

    """

    for asteroid in asteroids[1:]:
        minDeltaV = -1
        deltaV = []
        for tof in np.arange(tof0, tof1, step):
            for t in np.arange(t0, tf, step):
                # Coordinates of the first asteroid
                t1 = T_START.mjd2000 + t
                r1, v1 = asteroids[0].planetObject.eph(t1)

                # Coordinates of the second asteroid
                t2 = t1 + tof
                r2, v2 = asteroid.planetObject.eph(t2)

                # Compute Lambert solution
                l = pk.lambert_problem(
                    r1=r1,
                    r2=r2,
                    tof=t * pk.DAY2SEC,
                    mu=constants.MU_TRAPPIST,
                    cw=False,
                    max_revs=0
                )

                # Compute the delta-v necessary to go there
                # and match its velocity
                DV1 = [a - b for a, b in zip(v1, l.get_v1()[0])]
                DV2 = [a - b for a, b in zip(v2, l.get_v2()[0])]
                DV = np.linalg.norm(DV1) + np.linalg.norm(DV2)
                if DV < minDeltaV or minDeltaV == -1:
                    tMin = t
                    tofMin = tof
                    minDeltaV = DV
                deltaV.append(DV)

        x, y = np.meshgrid(np.arange(t0, tf, step), np.arange(tof0, tof1, step))
        z = np.array(deltaV).reshape(x.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.pcolormesh(x, y, z,
            norm=colors.LogNorm(z.min(), z.max()),
            cmap='viridis')
        ax.scatter(tMin,
                   tofMin,
                   s=75,
                   marker='+',
                   label='Optimize deltaV = {} at (t, tof) = ({}, {})'.format(
                          round(minDeltaV,2),
                          round(tMin,2),
                          round(tofMin,2)),
                    color='r')

        # Set legend center bottom
        ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2))
        ax.set_xlabel('Departure day (days)')
        ax.set_ylabel('Time of flight (days)')
        ax.set_title('Transfer window for asteroid {} from asteroid {}'.format(
            asteroid.asteroidId,
            asteroids[0].asteroidId
        ))

        # Show colorbar
        cbar = plt.colorbar(cs,
            extend='max')
        cbar.set_label(''r'$\Delta$''V (m/s)')
        plt.tight_layout()
        plt.savefig(f'{basename}.png')
        return tMin, tofMin, minDeltaV


if __name__ == '__main__':

    # Load asteroids
    lines = np.loadtxt('data/candidates.txt')
    asteroids = [Asteroid(line) for line in lines]

    # Plot mass histogram
    plot_mass_distribution(asteroids,
                           figurename='mass_histogram.png')

    # Plot asteroids by material
    plot_asteroids_by_material(asteroids,
                               materialType='Gold',
                               color='r',
                               figurename='gold.png')
    plot_asteroids_by_material(asteroids,
                               materialType='Platinum',
                               color='g',
                               figurename='platinum.png')
    plot_asteroids_by_material(asteroids,
                               materialType='Nickel',
                               color='b',
                               figurename='nickel.png')
    plot_asteroids_by_material(asteroids,
                               materialType='Propellant',
                               color='c',
                               figurename='propellant.png')

    plot_zenith_asteroids_by_material(asteroids,
                               materialType='Gold',
                               color='r',
                               figurename='gold_zenith.png')
    plot_zenith_asteroids_by_material(asteroids,
                               materialType='Platinum',
                               color='g',
                               figurename='platinum_zenith.png')
    plot_zenith_asteroids_by_material(asteroids,
                               materialType='Nickel',
                               color='b',
                               figurename='nickel_zenith.png')
    plot_zenith_asteroids_by_material(asteroids,
                               materialType='Propellant',
                               color='c',
                               figurename='propellant_zenith.png')

    # Plot asteroids
    plot_asteroids(asteroids)
    plot_zenith_asteroids(asteroids)
