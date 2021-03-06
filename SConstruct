#!/usr/bin/env python
#
# Build the code specified by SConscript files found under $PACKAGE_SOURCE.
# Put the output object and library files in the directory specified by $BUILD_BASE.
#
# Original author Rob Kutschke.
#
import os, re, string, sys

# Functions
from helpers import validateEnvironment
from helpers import validateDebugLevel
from helpers import defineMergeFlags
from helpers import defineExportedOSEnvironment
from helpers import locateSConscriptFiles
from helpers import dispatchSConscriptFiles

# Classes
from helpers import build_helper

# Check that some important environment variables have been set; exit on failure.
validateEnvironment()

# check we're not building in the source tree
if os.path.normpath(os.environ['PACKAGE_SOURCE']) == os.path.normpath(GetLaunchDir()):
    sys.exit('In-source builds are not allowed.\nExiting.')

# Get debug level from the environment and check its validity; exit on failure.
debugLevel = validateDebugLevel()

# Root directory of the source file tree.
sourceRoot = os.environ['PACKAGE_SOURCE']

# Root of the temporary space to be used to hold object files and libraries.
buildBase = os.environ['BUILD_BASE']

# Directory into which executables will be put.
bindir = buildBase+'/bin/'

# -I and -L paths for libraries that will be used.
includePath = [ sourceRoot,
                os.environ['ROOT_INC'] ]

linkPath    = [ os.environ['ROOT_LIB'],
                '#/lib'
              ]

rPath = [os.path.abspath(os.environ['ROOT_LIB']), os.path.join(buildBase,'lib/')]

# if sys.platform.startswith('linux'):
#     # if Linux
#     rPath += [Literal("'$$ORIGIN'/../lib")]
# elif sys.platform == 'darwin':
#     # if MacOS, use macOS relative rpath
#     rPath += [Literal('@executable_path/../lib')]

# Create and configure the scons environment that will be used during building.
env = Environment( CPPPATH   = [ includePath, ],
                   LIBPATH   = [ linkPath, ],
                   ENV       = defineExportedOSEnvironment(),
                   BUILDOPTS = [debugLevel],
                   BINDIR    = bindir,
                   RPATH     = rPath,
                   toolpath  = [os.path.join(sourceRoot, 'site_scons/site_tools')]
                  )

# Modify the environment: set compile and link flags.
SetOption('warn', 'no-fortran-cxx-mix')
env.MergeFlags( defineMergeFlags(debugLevel) )

# Define and register the rule for building dictionaries.
# sources are classes.h, classes_def.xml, 
# targets are dict.cpp, .rootmap and .pcm
# LIBTEXT is the library for the dict - not a target, only text for names
#genreflex = Builder(action=Action("export HOME="+os.environ["HOME"]+"; "+"genreflex ${SOURCES[0]} -s ${SOURCES[1]} $_CPPINCFLAGS -l $LIBTEXT -o ${TARGETS[0]} --fail_on_warnings --rootmap-lib=$LIBTEXT  --rootmap=${TARGETS[1]} $DEBUG_FLAG",genreflexcomstr))
#env.Append(BUILDERS = {'DictionarySource' : genreflex})

env.Tool('compilation_db')

# Generate a compile_commands.json
compileCommands = env.CompilationDatabase('compile_commands.json')
compileDb = env.Alias("compdb", compileCommands)

os.makedirs('lib', exist_ok=True)

# Make the environment visible to all SConscript files.
Export('env')

# Make an instance of the build_helper class and Export it.
# This makes it Import-able by the SConscript files.
Export('build_helper')

# Walk the directory tree to locate all SConscript files.
ss = locateSConscriptFiles(sourceRoot)

# Tell scons to do the work described by the SConscrip files.
dispatchSConscriptFiles( env, ss, sourceRoot, buildBase)

# This tells emacs to view this file in python mode.
# Local Variables:
# mode:python
# End:
