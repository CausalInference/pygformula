from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pygformula")
except PackageNotFoundError:
    __version__ = "unknown"

from .parametric_gformula import ParametricGformula
