#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
import pybind11
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys

# Package metadata
__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"
__description__ = "ROS 2 package implementing hyperdimensional computing for robust robotic control"

# Read long description from README
def read_long_description():
    """Read the long description from README.md"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()

# C++ extension modules
def get_extensions():
    """Get C++ extensions for the package"""
    extensions = []
    
    # HDC Core C++ binding
    hdc_core_sources = [
        "hdc_core/src/hypervector.cpp",
        "hdc_core/src/operations.cpp", 
        "hdc_core/src/memory.cpp",
        "hdc_core/src/encoding/spatial_encoder.cpp",
        "hdc_core/src/encoding/visual_encoder.cpp",
        "hdc_core/src/encoding/temporal_encoder.cpp",
        "hdc_core/src/reasoning/associative_memory.cpp",
        "hdc_core/src/reasoning/similarity.cpp",
        "python_bindings/hdc_core_bindings.cpp"
    ]
    
    hdc_core_ext = Pybind11Extension(
        "hdc_robot_controller._hdc_core",
        sources=hdc_core_sources,
        include_dirs=[
            "hdc_core/include",
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"{}"'.format(__version__))],
    )
    
    # Add CUDA support if available
    if os.environ.get("WITH_CUDA", "0") == "1":
        cuda_sources = [
            "hdc_core/src/cuda/bundle_kernel.cu",
            "hdc_core/src/cuda/bind_kernel.cu", 
            "hdc_core/src/cuda/similarity_kernel.cu",
        ]
        hdc_core_ext.sources.extend(cuda_sources)
        hdc_core_ext.define_macros.append(("WITH_CUDA", "1"))
        hdc_core_ext.libraries = ["cudart", "cublas"]
        hdc_core_ext.library_dirs = ["/usr/local/cuda/lib64"]
        hdc_core_ext.include_dirs.append("/usr/local/cuda/include")
    
    extensions.append(hdc_core_ext)
    
    return extensions

# Custom build command
class CustomBuildExt(build_ext):
    """Custom build extension to handle CUDA compilation"""
    
    def build_extensions(self):
        # Check for CUDA
        if os.environ.get("WITH_CUDA", "0") == "1":
            # Add CUDA-specific compiler flags
            for ext in self.extensions:
                if "cuda" in str(ext.sources):
                    ext.extra_compile_args = ["-O3", "-DWITH_CUDA"]
                    if sys.platform == "linux":
                        ext.extra_link_args = ["-lcudart", "-lcublas"]
        
        super().build_extensions()

# Development dependencies
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0", 
    "pytest-benchmark>=3.4.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "pre-commit>=2.15.0"
]

# ROS 2 dependencies (optional)
ros2_requirements = [
    "rclpy",
    "geometry-msgs",
    "sensor-msgs", 
    "nav-msgs",
    "tf2-py",
    "cv-bridge"
]

setup(
    name="hdc_robot_controller",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description=__description__,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/HDC-Robot-Controller",
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/HDC-Robot-Controller/issues",
        "Source": "https://github.com/danieleschmidt/HDC-Robot-Controller",
        "Documentation": "https://hdc-robot-controller.readthedocs.io",
    },
    packages=find_packages(include=["hdc_robot_controller*"]),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": CustomBuildExt},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "pybind11>=2.8.0",
    ],
    extras_require={
        "dev": dev_requirements,
        "ros2": ros2_requirements,
        "cuda": ["cupy>=9.0.0", "pycuda>=2021.1"],
        "viz": ["plotly>=5.0.0", "dash>=2.0.0"],
        "full": dev_requirements + ros2_requirements + ["cupy>=9.0.0", "plotly>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "hdc-demo=hdc_robot_controller.scripts.demo:main",
            "hdc-benchmark=hdc_robot_controller.scripts.benchmark:main",
            "hdc-visualizer=hdc_robot_controller.scripts.visualizer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="robotics, hyperdimensional-computing, hdc, machine-learning, fault-tolerance, ros2, control-systems",
    zip_safe=False,
    include_package_data=True,
    package_data={
        "hdc_robot_controller": [
            "*.yaml",
            "*.json", 
            "config/*.yaml",
            "launch/*.launch.py",
            "urdf/*.urdf",
            "meshes/*.stl",
            "models/*.pkl",
        ],
    },
    test_suite="tests",
)

# Post-installation message
print("""
🤖 HDC Robot Controller v{} installed successfully!

📚 Documentation: https://hdc-robot-controller.readthedocs.io
🐛 Report issues: https://github.com/danieleschmidt/HDC-Robot-Controller/issues  
💡 Examples: See examples/ directory for usage demonstrations

Optional features:
- CUDA acceleration: pip install hdc_robot_controller[cuda]
- ROS 2 integration: pip install hdc_robot_controller[ros2]  
- Visualization: pip install hdc_robot_controller[viz]
- All features: pip install hdc_robot_controller[full]

Quick start:
    import hdc_robot_controller as hdc
    hv = hdc.create_random_hypervector()
    print(f"Created hypervector with dimension {hv.dimension}")

Validate installation:
    python -c "import hdc_robot_controller; print('✅ OK' if hdc_robot_controller.validate_installation() else '❌ Failed')"
""".format(__version__))