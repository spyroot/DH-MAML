# This a unit test for 10 batch so all 10 passing, on GPU and CPU
python main.py --config configs/ant-vel.yaml --check_specs
#rm -rf models
# navigation
python main.py --config configs/2d-navigation.yaml --train --disable_wand --rpc_port 29400 --num_batches 10
# halfcheetah
python main.py --config configs/halfcheetah-vel.yaml --train --disable_wandb --rpc_port 29500 --num_batches 10
python main.py --config configs/halfcheetah-dir.yaml --train --disable_wandb --rpc_port 29600 --num_batches 10
# ant
python main.py --config configs/ant-dir.yaml --train --disable_wandb --rpc_port 29700 --num_batches 10
python main.py --config configs/ant-goal.yaml --train --disable_wandb --rpc_port 29800 --num_batches 10
python main.py --config configs/ant-vel.yaml  --train --disable_wandb --rpc_port 29900 --num_batches 10

# CPU only
# navigation
python main.py --config configs/2d-navigation.yaml --train --disable_wand --rpc_port 29100 --num_batches 10 --use-cpu
# halfcheetah
python main.py --config configs/halfcheetah-vel.yaml --train --disable_wandb --rpc_port 29200 --num_batches 10 --use-cpu
python main.py --config configs/halfcheetah-dir.yaml --train --disable_wandb --rpc_port 29300 --num_batches 10 --use-cpu
# ant
python main.py --config configs/ant-dir.yaml --train --disable_wandb --rpc_port 29350 --num_batches 10 --use-cpu
python main.py --config configs/ant-goal.yaml  --train --disable_wandb --rpc_port 29330 --num_batches 10 --use-cpu
python main.py --config configs/ant-vel.yaml  --train --disable_wandb --rpc_port 29652 --num_batches 10 --use-cpu