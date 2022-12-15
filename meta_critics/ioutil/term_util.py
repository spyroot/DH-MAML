class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_red(msg: str):
    print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")


def print_blue(msg: str):
    print(f"{bcolors.OKBLUE}{msg}{bcolors.ENDC}")


def print_green(msg: str):
    print(f"{bcolors.OKGREEN}{msg}{bcolors.ENDC}")


def red_str(msg: str) -> str:
    return f"{bcolors.WARNING}{msg}{bcolors.ENDC}"


def blue_str(msg: str) -> str:
    return f"{bcolors.OKBLUE}{msg}{bcolors.ENDC}"


def green_str(msg: str) -> str:
    return f"{bcolors.OKGREEN}{msg}{bcolors.ENDC}"
