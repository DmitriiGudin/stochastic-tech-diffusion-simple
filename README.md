# Prerequisites

Python 3.12


# Installation

1. Make any parameter changes in config.py. The default parameters were used in the dissertation.

2. Run
```
python run.py -- all
```

The results will appear in the *figures* folder. To replicate the boundary effects curve in 1D, change the value of *dim* (line 78) in *config.py* from *2* to *1*, then run
```
python run.py -- mean
```
