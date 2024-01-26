def print_duration(activity: str, start_time: float, end_time: float):
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Time for {activity}: {hours} hours, {minutes} minutes, and {seconds} seconds\n")