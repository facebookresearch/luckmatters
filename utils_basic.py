from datetime import datetime

def signature():
    return str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")

def sig():
    return datetime.now().strftime("%m%d%y_%H%M%S_%f")

