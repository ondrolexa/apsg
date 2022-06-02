# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from apsg.config import apsg_conf
from apsg.plotting._stereonet import StereoNet


def zijderveld_plot(core, kind="geo"):
    def onpick(core, event, fig):
        fig.suptitle("{}".format(core.steps[event.ind[0]]))
        fig.canvas.draw()

    data = getattr(core, kind)
    N, E, Z = np.array(data).T
    N0, E0, Z0 = data[0]
    fig, ax = plt.subplots(facecolor="white", figsize=apsg_conf["figsize"])
    ax.plot(E, N, "b-", label="Horizontal")
    ax.plot(E0, N0, "b+", markersize=14)
    ax.plot(E, N, "bo", picker=5)
    ax.plot(E, -Z, "g-", label="Vertical")
    ax.plot(E0, -Z0, "g+", markersize=14)
    ax.plot(E, -Z, "go", picker=5)
    fig.canvas.mpl_connect("pick_event", lambda event: onpick(event, fig))
    mx = np.max(np.abs(ax.axis()))
    ax.axis([-mx, mx, -mx, mx])
    ax.set_aspect(1)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_position("zero")
    t = ax.xaxis.get_ticklocs()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # t = ax.xaxis.get_ticklocs()
    # ax.xaxis.set_ticks(t[t != 0])
    # t = ax.yaxis.get_ticklocs()
    # ax.yaxis.set_ticks(t[t != 0])
    ax.set_title("{} {}".format(core.site, core.specimen), loc="left")
    plt.legend(title="Unit={:g}A/m".format(t[1] - t[0]))
    plt.tight_layout()
    plt.show()


def demag_plot(core):
    fig, ax = plt.subplots(figsize=apsg_conf["figsize"])
    ax.plot(core.nsteps[0], core.MAG[0] / core.MAG.max(), "k+", markersize=14)
    ax.plot(core.nsteps, core.MAG / core.MAG.max(), "ko-")
    ax.set_ylabel("M/Mmax")
    ax.set_title("{} {} (Mmax = {:g})".format(core.site, core.specimen, core.MAG.max()))
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid()
    plt.show()


def stereo_plot(core, kind="geo", **kwargs):
    data = getattr(core, kind)
    tt = {
        "V": "Specimen coordinates",
        "geo": "Geographic coordinates",
        "tilt": "Tilted coordinates",
    }
    s = StereoNet(
        title="{} {}\n{}".format(core.site, core.specimen, tt[kind]), **kwargs
    )
    for f1, f2 in zip(data[:-1], data[1:]):
        s.arc(f1, f2, "k:")
    s.vector(data[0], "k+", markersize=14)
    s.vector(data, "ko")
    s.show()
