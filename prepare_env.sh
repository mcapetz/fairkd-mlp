# This code in part draws from: https://github.com/snap-research/graphless-neural-networks
# Please see the license here: https://github.com/snap-research/graphless-neural-networks/blob/main/LICENSE
# Changes were made to the original code.

#!/bin/bash

conda create -y -n glnn python=3.6.9
eval "$(conda shell.bash hook)"
conda activate glnn

pip install --no-cache-dir -r requirements.txt
