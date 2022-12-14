conda activate meta_critic
#rm -rf models
python main.py --config configs/2d-navigation.yaml --train --disable_wand --rpc_port 29400
python main.py --config configs/halfcheetah-vel.yaml --train --disable_wandb --rpc_port 29500
python main.py --config configs/halfcheetah-dir.yaml --train --disable_wandb --rpc_port 29600
