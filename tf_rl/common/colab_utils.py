import os
import shutil


def copy_dir(src, dst, symlinks=False, ignore=None, verbose=False):
    """ copy the all contents in `src` directory to `dst` directory """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if verbose:
            print("From:{}, To: {}".format(s, d))
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def delete_files(folder):
    """ delete the all contents in `folder` directory """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        print("{} has been refreshed".format(folder))
        os.makedirs(folder)
