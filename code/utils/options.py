import os
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
#from . import appdirs

cwd = os.path.dirname(__file__)
#userdir = appdirs.user_data_dir("pycortex", "JamesGao")
#usercfg = os.path.join(userdir, "options.cfg")
usercfg = os.path.join(cwd, "options.cfg")

config = configparser.ConfigParser()
config.readfp(open(os.path.join(cwd, 'defaults.cfg')))

# the config.read(usercfg) call, even though it is an if statement,
# reads in user-specific configuration file and overwrites defaults
if len(config.read(usercfg)) == 0:
    #os.makedirs(userdir)
    with open(usercfg, 'w') as fp:
        config.write(fp)