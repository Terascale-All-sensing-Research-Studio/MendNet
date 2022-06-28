# Make libs
if [ ! -d "/path/to/dir" ] 
then
    mkdir libs
fi
cd libs &&

# Install cython first
pip install cython cmake

# Clone inside mesh
git clone https://github.com/nikwl/inside_mesh.git
cd inside_mesh

# Install inside mesh
python setup.py build_ext --inplace &&
pip install .

cd .. && \
    git clone https://github.com/davidstutz/mesh-fusion.git && \
    cd mesh-fusion && \
    cd libfusioncpu && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd .. && \
    python setup.py build_ext --inplace && \
    pip install . && \
    cd .. && \
    cd librender && \
    python setup.py build_ext --inplace && \
    mv pyrender.cpython-36m-x86_64-linux-gnu.so pyrender.so && \
    pip install .