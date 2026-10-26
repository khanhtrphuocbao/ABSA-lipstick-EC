"""Registered ABSA training primitives; importing this package never loads data or fits a model."""

from .targets import encode_batch

__all__ = ['encode_batch']
