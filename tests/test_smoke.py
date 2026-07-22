def test_package_imports():
    """The package imports and exposes a version string — proves the scaffold is installable."""
    import ctxdiff
    assert isinstance(ctxdiff.__version__, str)
