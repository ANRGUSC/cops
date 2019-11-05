# Graph connectivity for intermittent connection in multi-agent networks

This package is developed for Python 3.7.

# Setup instructions for Debian-like environments

1. Install dependencies
```bash
sudo apt install python3 graphviz ffmpeg
```

2. (Optional) Set up and activate a virtual environment
```bash
virtualenv -p python3 ~/.venvs/pyenv
source ~/.venvs/pyenv/bin/activate
```

3. Install [Gurobi](https://www.gurobi.com/) and/or [Mosek](https://www.mosek.com/), as well as their Python interface packages
```bash
# Install python gurobi interface
cd gurobi_path/linux64
python setup.py install
python -c 'import gurobipy'   # should not fail

# Install python mosek interface
cd mosek_path/tools/platform/linux64x86/python/3
python setup.py install
python -c 'import mosek'   # should not fail
```

4. Install python dependencies and cops
```bash
cd cops_path/
pip install -r requirements.txt
python setup.py install
nosetests  # optional: run tests
```
