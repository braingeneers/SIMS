"""Optional integrations for scsims.

These modules are not imported by ``scsims/__init__.py`` and may have
extra dependencies (boto3, etc.). Install them via the appropriate
``pip install scsims[<extra>]`` extra.
"""

try:
    from .upload import UploadCallback  # noqa: F401
except ModuleNotFoundError:
    # boto3 not installed; UploadCallback unavailable.
    pass
