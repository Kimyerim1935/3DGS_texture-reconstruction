"""Standalone SH appearance restoration for Graphdeco-format Gaussian PLYs."""


def restore_texture(*args, **kwargs):
    from .restore import restore_texture as restore
    return restore(*args, **kwargs)


__all__ = ["restore_texture"]
