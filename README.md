# STAT3007 Project
STAT3007 2024 Semester 1 Group Project

## Setting up the environment
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

## Using getafix 

To access the cluster, ssh in with 
```
ssh sXXXXXXX@getafix.smp.uq.edu.au -p 2022
```
Or you can add this to your `.ssh/config` file.  
```
Host getafix
  HostName getafix.smp.uq.edu.au
  User sXXXXXXX
  Port 2022
```

You will be asked for your UQ account password and a Duo code. The Duo code will not be a popup, rather you must open the app and use Show the passcode. Once this is done, you will be on the login node and able to view your home directory. 

### Submitting jobs on getafix

You should not attempt to run complex code on the login node, as it does not have much compute. Rather, you should be submitting a job. The cluster uses the slurm job manager to handle jobs. Some useful slurm commands include
```
sbatch [slurm file] - submit a job to the cluster
squeue - view information about jobs in the queue
scancel [job number] - cancel a job
sinfo - ivew information about nodes / partitions
```

### Slurm file

The slurm file contains information for how jobs are executed. It is basically just a text file, though we can give it a `.sbatch` file extension to make it clear its a slurm file. 

We also need to load modules in the slurm file. The one we need is `module load pytorch`. After that, we can specify what code to execute. 
```slurm
#!/bin/bash

#SBATCH --job-name=[job-name] # name of the job
#SBATCH --time=[D-HH:MM:SS] # maximum time for job - it will get cancelled after this much 15:28
#SBATCH --mem=[xG] # allocated x GB of memory
#SBATCH --output=[output_file.txt] # sends any print output to this file, which is sent to you
#SBATCH --partition=[partition] # which partition of the cluster to use (we will use gpu)
#SBATCH --gres=gpu:[n] # allocate n gpus to the task

# load modules

module load pytorch 

# execute code here
python3 pythonscript.py

```

Please navigate to `/test` and try running `sbatch test.sbatch`. The output file will appear in `torch_test.txt`


