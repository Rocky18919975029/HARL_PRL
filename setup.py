from setuptools import find_packages, setup

setup(
    name="harl",
    version="1.1.0",
    author="PKU-MARL",
    description="Critic-guided predictive representations for policy-gradient MARL",
    url="https://github.com/Rocky18919975029/HARL_PRL",
    packages=find_packages(),
    package_data={"harl": ["configs/algos_cfgs/*.yaml", "configs/envs_cfgs/*.yaml"]},
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0,<2.6",
        "numpy>=1.21,<2",
        "absl-py",
        "cloudpickle",
        "tqdm",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
