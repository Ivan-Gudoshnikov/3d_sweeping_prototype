# _mkl_bootstrap.py
import os
import threading


_mkl_loaded = threading.Event()


def _load_mkl_win() -> None:  # pragma: no cover
    r"""Load, if necessary, Intel's Math Kernel Library (MKL).

    First, why might we want MKL, and second, when is it necessary to load it
    now?

    CVXOPT requires MKL to implement BLAS, which is a set of industry-standard
    linear algebra subroutines. CVXOPT only requires MKL on Windows because
    OpenBLAS is generally available on Linux and macOS comes with its own
    Accelerate Framework implementation of BLAS. Intel provides MKL as a Wheel
    on PyPI, which Pip installs as part of installing CVXOPT. See `#127
    <https://github.com/cvxopt/cvxopt/issues/127>`_.

    Intel's MKL wheel places a bunch of DLLs in the
    :file:`{exec_prefix}\Library\bin` directory, where :file:`{exec_prefix}` is
    :const:`sys.exec_prefix`. It chooses this location because the Conda package
    manger, which is popular in the scientific community but is not standard in
    the larger Python community, `uses this directory on Windows for DLLs
    <https://conda.io/docs/user-guide/tasks/build-packages/use-shared-libraries.html#shared-libraries-in-windows>`_.
    If you run Conda's version of Python, then the MKL wheel works out of the
    box. If you run standard Python, you're in trouble.

    This is because Windows loads DLLs from the system PATH environment
    variable. :file:`{exec_prefix}\Library\bin` is not on the PATH by default.
    We could add it to the PATH when this module loads, but that's a side effect
    of loading a library module that client code should not have to plan around
    (e.g., client code might load this module and then try to run a system
    command and be surprised to end up running a different command only because
    it's in :file:`{exec_prefix}\Library\bin`).

    Fortunately, Windows only loads DLLs once. So all we have to do is load MKL
    *before* importing CVXPY (which in turn imports CVXOPT) with
    :file:`{exec_prefix}\Library\bin` on the PATH, and then remove it from the
    PATH. That's what this function does.

    :raises ImportError:
        If MKL cannot be found.
    """
    # Only needed on Windows.
    if os.name != 'nt':
        return
    # Only needed once per process
    if  _mkl_loaded.is_set():
        return
    # cdll vs windll doesn't matter because we're not actually calling any
    # functions. We use cdll here because Mypy complains about windll's
    # unavailability on non-Windows systems.
    from ctypes import cdll
    import sys
    # Use sys.exec_prefix because that's the home of platform-dependent files.
    lib_bin = os.path.join(sys.exec_prefix, 'Library', 'bin')
    # These are the two DLLs that Windows complains about if not preloaded.
    libs = ['mkl_rt', 'mkl_intel_thread']
    libs = [os.path.join(lib_bin, name + os.extsep + 'dll') for name in libs]
    old_path = os.environ['PATH']
    os.environ['PATH'] = os.pathsep.join([lib_bin, old_path])
    for lib in libs:
            try:
                cdll.LoadLibrary(lib)
            except OSError as e:
                raise NameError("Cannot find Intel's Math Kernel Library (mkl)")
            else:
                _mkl_loaded.set()
            finally:
                os.environ['PATH'] = old_path