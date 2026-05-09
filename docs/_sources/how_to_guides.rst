How-to Guides
#############

Installing the library
**********************

There are multiple ways of installing the library. The recommended way is to use `poetry` package manager, which will create virtual environment for the project and install all required dependencies.

Alternative way is to install the library using `pip` package manager within already existing virtual environment.

Installation the library with `poetry` from provided source code
================================================================


1. Installing  and initializing `poetry` through `pipx`

.. code-block::
    :caption: Installing pipx

        > python -m pip install --user pipx 
        > python -m pipx ensurepath


.. code-block::
    :caption: Installing poetry

        > pipx install poetry

2. Check if `poetry` is successfully installed

.. code-block::
    :caption: Check poetry version

        > poetry --version

3. Install project's packets and documentation builder packets with `poetry` 

.. code-block::
    :caption: Specify that `poetry` will work with python version 3.13 

        > poetry env use 3.13

    The previous command will create virtual environment based on `python3.11` in subdirectory `/.venv` 

.. code-block::
    :caption: Install dependencies (and documentation dependencies) with `poetry`

        > poetry install --with docs


Installation the library with `pip` from provided source code
=============================================================

1. Create and activate virtual environment

.. code-block::
    :caption: Create and activate virtual environment

        > python -m venv venv
        > source venv/bin/activate      # On Linux
        > .\venv\Scripts\activate       # On Windows

2. Install required dependencies from `requirements.txt` file

.. code-block::
    :caption: Install dependencies with pip

        > pip install -r requirements.txt
        > pip install -r requirements-docs.txt

Running of all the unit tests within developed library
******************************************************

Unit tests are located within `tests` subdirectory of each application. To run all the unit tests, the following commands can be used.

.. code-block::
    :caption: Run all unit tests within project

        > python -m unittest

Obtaining coverage analysis of all unit tests within library can be done with `coverage` package. The following commands can be used depending on the way how the library was installed.


.. code-block::
    :caption: Obtain coverage analysis of tests within project

        > python -m coverage run -m unittest
        > python -m coverage report


Building documentation for the library
**************************************

1. Build documentation sources into `/documentation/source` folder from `python` source files 

.. code-block::
    :caption: Build documentation sources

        > sphinx-apidoc -o documentation/source/ uo
        > sphinx-apidoc -o documentation/source/ opt


2. Change current directory to `/documentation` 

.. code-block::
    :caption: Change directory

        > cd documentation

3. Clean previously builded HTML documentation 

.. code-block::
    :caption: Clean HTML documentation 

        /documentation> ./make clean html

4. Build HTML documentation from `/documentation/source` directory. Created documentation is within `/documentation/build/html` directory. 

.. code-block::
    :caption: Build HTML documentation 

        /documentation> ./make html

5. Generated documentation, that is in folder `/documentation/build/html` should be then copied into folder `/docs`.

.. code-block::
    :caption: Copy generated HTML documentation 

        /documentation> cp build/html/*.* ../docs


