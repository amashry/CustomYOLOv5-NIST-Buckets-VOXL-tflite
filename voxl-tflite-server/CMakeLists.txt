cmake_minimum_required(VERSION 3.3)

project(voxl-tflite-server)

# James' standard list of cmake flags
set(CMAKE_C_FLAGS "-std=gnu99 -Wno-unused-parameter -Wall -Wextra -Wuninitialized \
	-Wunused-variable -Wdouble-promotion -Wmissing-prototypes \
	-Wmissing-declarations -Werror=undef -Wno-unused-function ${CMAKE_C_FLAGS}")

set(CMAKE_CXX_FLAGS "-std=gnu++11 -std=c++11 -Wno-unused-parameter -Wall -Wextra -Wuninitialized -Wno-attributes \
    -Wunused-variable -Wno-double-promotion -Wno-missing-field-initializers \
    -Werror=undef -Wno-unused-function ${CMAKE_CXX_FLAGS}")

# for VOXL, install 64-bit libraries to lib64
set(LIB_INSTALL_DIR /usr/lib64)

# include each subdirectory, may have others in example/ or lib/ etc
add_subdirectory (src)
