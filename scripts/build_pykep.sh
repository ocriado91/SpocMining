# Clone pykep repository if it hasn't been cloned previously.
CURRENT_DIRECTORY=$(pwd)
if [ ! -d "${CURRENT_DIRECTORY}"/pykep_source ]; then git clone https://github.com/esa/pykep pykep_source; fi

## Build Keplerian Toolbox library
# Create build folder and move into it
mkdir "${CURRENT_DIRECTORY}"/pykep_source/build/
cd "${CURRENT_DIRECTORY}"/pykep_source/build/

# Run CMake
cmake -DBoost_NO_BOOST_CMAKE=ON \
    -DPYKEP_BUILD_KEP_TOOLBOX=yes \
    -DPYKEP_BUILD_PYKEP=no \
    -DPYKEP_BUILD_SPICE=yes \
    -DPYKEP_BUILD_TESTS=yes \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_PREFIX_PATH=/usr/local/ \
    -DCMAKE_BUILD_TYPE=Release \
    ../;

# And execute make
cmake  --build . --target install

## Build pykep package
# Create build pykep folder and move into it
mkdir "${CURRENT_DIRECTORY}"/pykep_source/build_pykep
cd "${CURRENT_DIRECTORY}"/pykep_source/build_pykep

# Run CMake with pykep building flags enabled
cmake -DBoost_NO_BOOST_CMAKE=ON \
    -DPYKEP_BUILD_KEP_TOOLBOX=no \
    -DPYKEP_BUILD_PYKEP=yes \
    -DPYKEP_BUILD_TESTS=no \
    -DPYTHON_EXECUTABLE=/usr/bin/python3\
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_PREFIX_PATH=/usr/local/ \
    -DCMAKE_BUILD_TYPE=Release \
    ../;
# And execute make
make -j4 install