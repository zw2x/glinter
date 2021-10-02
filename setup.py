from setuptools import setup, find_packages

setup(
    name="glinter",
    version='0.0.1',
    description="Graph Learning of INTER-protein contacts",
    setup_requires=[
        "setuptools>=18.0",
    ],
    install_requires=[
        "numpy",
        "tqdm",
        "biopython",
        "matplotlib",
        "trimesh",
        "scipy",
    ],
    packages=find_packages(
        exclude=[
            "examples",
            "examples.*",
            "scripts",
            "scripts.*",
            "tests",
            "tests.*",
        ]
    ),
    test_suite="tests",
    zip_safe=False,
)
