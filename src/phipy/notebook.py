try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None


def setup(
    autoreload: bool = True,
    autoreload_mode: str = "3",
    matplotlib: str | None = "inline",
    display_width: str | None = "100%",
) -> None:
    """Configure common IPython notebook defaults; no-op outside IPython."""
    ip = get_ipython() if get_ipython else None
    if ip is None:
        return

    if autoreload:
        try:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", autoreload_mode)
        except Exception:
            pass

    if matplotlib:
        try:
            ip.run_line_magic("matplotlib", matplotlib)
        except Exception:
            pass

    if display_width:
        try:
            from IPython.display import display, HTML

            display(
                HTML(
                    "<style>.container { width:"
                    f" {display_width} !important; }}</style>"
                )
            )
        except Exception:
            pass
