try:
    import paderbox.database
except:
    raise NotImplementedError(
        "These examples are depending on our internal database structure"
        "at the moment."
        "Trying to execute them anyway may take considerable "
        "effort on your behalf."
)