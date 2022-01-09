"""Utilities for handling I/O and pathing."""
import glob
import os


class BadPathError(IOError):
    """Raise when something is funky with a user given path."""


def resolve_path(path, check_if_exists=False, expand_regex=False):
    """Returns an absolute path, resolving all expansions.

    Parameters
    ----------
    path : str | List[str]
        A possibly relative path to a file or directory.
        If a list, expand_regex must be true. All patterns
        are concatenated into the same list.
    check_if_exists : bool, optional
        If True, raises an error if the resolved path does not exist.
    expand_regex : bool, optional
        If True, then we use glob to expand this pattern.

    Returns
    -------
    path : str
        If expand_regex=False
    paths : str
        If expand_regex=True

    Raises
    ------
    BadPathError
        If `check_if_exists` is true and the resolved path does not exist.

    """

    # first handle lists
    if isinstance(path, list):
        assert expand_regex
        all_paths = []
        for pattern in path:
            paths = resolve_path(
                pattern, check_if_exists=check_if_exists, expand_regex=True
            )
            all_paths.extend(paths)
        return all_paths

    # Handle relative path to catkin packages
    if path.startswith("pkg://"):
        path = path.lstrip("pkg://")
        pkg_name = path.split("/")[0]
        rel_path_name = path.lstrip(f"{pkg_name}/")
        try:
            import rospkg

            rospack = rospkg.RosPack()
            pkg_dir = rospack.get_path(pkg_name)
        except:
            import git

            # try searching relative to root of this repo
            git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
            git_root = git_repo.git.rev_parse("--show-toplevel")
            pkg_dir = os.path.join(git_root, pkg_name)
        path = os.path.join(pkg_dir, rel_path_name)

    path = os.path.abspath(os.path.expanduser(path))
    if check_if_exists and not os.path.exists(path):
        raise BadPathError(f"path {path} does not exists.")

    if expand_regex:
        paths = glob.glob(path)
        return paths
    else:
        return path


def ensure_dir(path):
    """Create a directory if it does not exist already.

    Parameters
    ----------
    path : str
        The path to a directory that is created (if necessary).

    Returns
    -------
    path : str
        The absolute path to the given directory, which now exists
        for sure.

    """
    path = resolve_path(path)

    # If the path exists, make sure it is indeed a directory.
    if os.path.exists(path):
        if os.path.isfile(path):
            raise BadPathError(
                f"You tried to ensure_dir {path}. "
                + "This path already exists but is a file."
            )

        return path

    # Otherwise create the directory
    else:
        os.makedirs(path)
        return path


def ensure_nonexistent_and_create_dir(path, create_parent_dir=False):
    """Make sure given path does not exist. Create appropriate directory.

    Parameters
    ----------
    path : str
        The path to a file or directory.
    create_parent_dir : bool, optional
        If false, we assume the given path is a directory, and create that
        directory. Else, we assume it's a file. If the file exists, we raise
        an error. Otherwise, we create its parent directory only if the
        parent directory does not already exist.

    Returns
    -------
    dir_name : str
        The absolute path of the created directory (or the file if
        `create_parent_dir=True`).

    Raises
    ------
    PathAlreadyExistsError
        If the given path already exists.

    """
    path = resolve_path(path)

    if os.path.exists(path):
        raise BadPathError(
            f"Path {path} already exists. Crashing now to avoid overwriting data."
        )

    if create_parent_dir:
        ensure_dir(os.path.dirname(path))
    else:
        os.makedirs(path)

    return path
