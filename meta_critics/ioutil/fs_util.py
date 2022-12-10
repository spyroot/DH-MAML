from pathlib import Path

from meta_critics.running_spec import RunningSpec


def resole_primary_dir(path_to_dir: str) -> str:
    """
    :param path_to_dir:
    :return:
    """
    log_dir = path_to_dir.strip()
    if log_dir.startswith("~"):
        log_dir = str(Path.home()) + log_dir[1:]

    log_file_path = Path(log_dir).expanduser().resolve()

    if not log_file_path.exists():
        print(f"{log_dir} not found.")
        raise FileNotFoundError(f"Error: dir {log_dir} not found.")

    if not log_file_path.is_dir():
        print(f"{log_dir} must be directory.")
        raise FileNotFoundError(f"Error {log_dir} must be directory not a file.")

    return str(log_file_path)


def resole_primary_from_spec(spec: RunningSpec) -> str:
    return resole_primary_dir(spec.get("log_dir"))
