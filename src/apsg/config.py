# Default module settings.

apsg_conf = dict(
    notation="dd",  # notation geological measurements (dd or rhr)
    vec2geo=False,  # repr Vector3 using notation
    ndigits=3,  # Round to ndigits in repr
    figsize=(8, 6),  # Default figure size
    stereonet_default_kwargs=dict(
        kind="equal-area",
        overlay_position=(0, 0, 0, 0),
        rotate_data=False,
        minor_ticks=None,
        major_ticks=None,
        overlay=True,
        overlay_step=15,
        overlay_resolution=181,
        clip_pole=15,
        hemisphere="lower",
        grid_type="gss",
        grid_n=2000,
    ),
    default_point_kwargs=dict(
        alpha=None, color=None, mec=None, mfc=None, ls="none", marker="o", mew=1, ms=6,
    ),
    default_pole_kwargs=dict(
        alpha=None, color=None, mec=None, mfc=None, ls="none", marker="o", mew=1, ms=6,
    ),
    default_vector_kwargs=dict(
        alpha=None, color=None, mec=None, mfc=None, ls="none", marker="o", mew=2, ms=6,
    ),
    default_great_circle_kwargs=dict(
        alpha=None, color=None, ls="-", lw=1.5, mec=None, mew=1, mfc=None, ms=2,
    ),
    default_scatter_kwargs=dict(
        alpha=None,
        s=None,
        c=None,
        linewidths=1.5,
        marker=None,
        cmap=None,
        legend=False,
        num="auto",
    ),
    default_cone_kwargs=dict(
        alpha=None, color=None, ls="-", lw=1.5, mec=None, mew=1, mfc=None, ms=2,
    ),
    default_pair_kwargs=dict(
        alpha=None,
        color=None,
        ls="-",
        lw=1.5,
        mec=None,
        mew=1,
        mfc=None,
        ms=4,
        line_marker="o",
    ),
    default_fault_kwargs=dict(
        alpha=None, color=None, ls="-", lw=1.5, mec=None, mew=1, mfc=None, ms=2,
    ),
    default_hoeppner_kwargs=dict(
        alpha=None, color=None, mec=None, mfc=None, ls="none", marker="o", mew=1, ms=5,
    ),
    default_quiver_kwargs=dict(
        color=None, width=2, headwidth=5, pivot="mid", units="dots",
    ),
    default_contourf_kwargs=dict(
        alpha=None,
        antialiased=True,
        cmap="Greys",
        levels=6,
        clines=True,
        linewidths=1,
        linestyles=None,
        colorbar=False,
        trim=True,
        sigma=None,
    ),
)
