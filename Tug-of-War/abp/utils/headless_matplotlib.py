# When running any matplotlib-based code in a non-graphical environment,
# you may encounter errors such as the following:
#
# ImportError: No module named '_tkinter', please install the python3-tk package
#
# The following code checks if your Python program is being run in a text-based
# environment (eg. an SSH console) and configures Matplotlib accordingly
import os
if not os.environ.get('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
