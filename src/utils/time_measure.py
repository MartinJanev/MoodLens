def measure_time(sec):
    """
    Format seconds into a human-readable string of hours, minutes, and seconds.
    :param sec: Time in seconds.
    :return: Formatted string like "1h 23m 45s" or "45s".
    """
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else (
        f"{m}m {s}s" if m else f"{s}s")