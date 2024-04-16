import matplotlib as mpl
import matplotlib.pyplot as plt


def init_style(fontsize: int = 18, font: str = "CMU Serif") -> None:
    """
    Initialize matplotlib style settings.

    This function sets various style parameters using the `style` module to customize the look and feel of matplotlib
    plots. The default settings are suitable for producing high-quality figures in research articles and presentations.

    Parameters
    ----------
    fontsize : int, optional
        Font size to be used in figures. Default is 18.
    font : str, optional
        Font family to be used in figures. Default is "CMU Serif".

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from brainvistools.style import init_style
    >>> init_style(fontsize=20, font="Times New Roman")
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> plt.show()
    """
    plt.style.use(
        {
            "figure.facecolor": "white",  # background color of figures
            "text.usetex": True,  # use LaTeX to render text
            "font.family": font,  # font family for all text
            "font.weight": "bold",  # use bold fonts
            "font.size": fontsize,  # default font size
            "savefig.dpi": 800,  # resolution for saved figures
            "savefig.facecolor": "None",  # don't save background color of figures
        }
    )

    # plt.rcParams.update(
    #     {
    #         "figure.facecolor": "white",
    #         "pgf.texsystem": "pdflatex",
    #         "text.usetex": True,
    #         "font.family": font,
    #         "font.weight": "bold",  # bold fonts
    #         "font.size": fontsize,
    #         "savefig.dpi": 800,  # higher resolution output.
    #         "savefig.facecolor": "None",
    #     }
    # )


def mpl_style(style="classic"):
    """
    Set the current matplotlib style.

    Parameters
    ----------
    style : str, optional
        Name of the style to use. Default is "classic".

    Returns
    -------
    None
    """
    mpl.style.use(style)


def set_fontsize(fontsize=25):
    """
    Set the font size to be used in figures.

    Parameters
    ----------
    fontsize : int, optional
        Font size to be used in figures. Default is 25.

    Returns
    -------
    None
    """
    plt.rcParams["font.size"] = fontsize


def set_font(font="CMU Serif"):
    """
    Set the font family to be used in figures.

    Parameters
    ----------
    font : str, optional
        Font family to be used in figures. Default is "CMU Serif".

    Returns
    -------
    None
    """
    plt.rcParams["pgf.texsystem"] = "pdflatex"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = font
