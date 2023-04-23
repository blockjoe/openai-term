import os
import glob
from typing import Generator, Optional


def get_latest_filename(
    file_dir: str, file_ext: str, data_dir: Optional[str] = None
) -> str:
    latest = max((int(fname) for fname in iter_filenames(file_dir, file_ext, data_dir=data_dir)))
    return os.path.join(file_dir, str(latest) + file_ext)


def get_oldest_filename(
    file_dir: str, file_ext: str, data_dir: Optional[str] = None
) -> str:
    oldest = min((int(fname) for fname in iter_filenames(file_dir, file_ext, data_dir=data_dir)))
    return os.path.join(file_dir, str(oldest) + file_ext)

def iter_filepaths(
    file_dir: str, file_ext: str, data_dir: Optional[str] = None
) -> Generator[str, None, None]:
    if not file_ext.startswith("."):
        file_ext = "." + file_ext
    if data_dir:
        file_dir = os.path.join(data_dir, file_dir)
    glob_pattern = os.path.join(file_dir, "*" + file_ext)
    yield from sorted(glob.iglob(glob_pattern), key=os.path.getmtime, reverse=True)

def iter_filenames(
    file_dir: str, file_ext: str, data_dir: Optional[str] = None
) -> Generator[str, None, None]:
    for fpath in iter_filepaths(file_dir, file_ext, data_dir=data_dir):
        entry, _ = os.path.splitext(os.path.basename(fpath))
        yield entry

