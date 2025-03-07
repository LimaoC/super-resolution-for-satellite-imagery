# STAT3007 Project
STAT3007 2024 Semester 1 Group Project

## Setting up the environment
### General environment
*Note you may need to substitue "python" with "python3" or "py" in the following commands depending on your Python installation*

After cloning the repo navigate to its location, for example 
```bash
cd ~/GitHub/stat3007-project
```
Next create a virtual environment.
```bash
python -m venv myenv
```
Activate the virtual environment. For windows
```bash
source myenv/Scripts/activate
```
For POSIX
```bash
source myenv/bin/activate
```
For windows only, you can install requirements with the following.
```bash
pip install -r requirements.txt
```
Install our project package in edit mode
```bash
pip install -e .
```
Start a Jupyter Lab server
```bash
jupyter lab
```
You should now be able to run the notebook at `super_resolution/notebooks/example_notebook.ipynb`.

To deactivate the virtual environment when done, use
```bash
deactivate
```
