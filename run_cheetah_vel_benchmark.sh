python main.py --config configs/halfcheetah-vel.yaml --benchmark --train --num_batches 10
python main.py --config configs/halfcheetah-vel.yaml --benchmark --train --workers 2 --num_batches 10
python main.py --config configs/halfcheetah-vel.yaml --benchmark --train --workers 4 --num_batches 10
python main.py --config configs/halfcheetah-vel.yaml --benchmark --train --workers 4 --num_worker_threads 4 --num_batches 10
python main.py --config configs/halfcheetah-vel.yaml --benchmark --train --workers 4 --num_worker_threads 16 --num_batches 10
python main.py --config configs/halfcheetah-vel.yaml --benchmark --train --workers 2 --num_worker_threads 4 --num_batches 10
