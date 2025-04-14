from setuptools import setup

setup(
    name="housinglib",
    version="0.0.1",
    description="Housing Price Prection Code Library",
    long_description="This library contains the functions like loading data \
        and creating features and training the data.",
    author="Upasana Saha",
    py_modules=["housinglib"],
    package_dir={"": "src/housinglib"},
)