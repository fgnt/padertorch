try:
    import padercontrib.database
except:
    import warnings
    warnings.warn(
        "These examples are depending on our internal database structure "
        "at the moment. "
        "Trying to execute them anyway may take considerable "
        "effort on your behalf."
    )
