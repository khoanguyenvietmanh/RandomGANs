# old training code at: /train/*.py
python -m src.train.hgan --exp_name hgan32
python -m src.train.rfgan --exp_name rfgan32 --weights experiments/rfgan32/models/model_96000.pt --it 96000
python -m src.train.gan1 --exp_name gan1_32

# new training code at: train/py
python -m src.train --exp_name 'gan1_ring' --nepochs '501' --nheads '1' --type 'gan1' --data_type 'grid'
python -m src.train --exp_name 'hgan_ring' --nepochs '501' --nheads '5' --type 'hgan' --data_type 'grid'
python -m src.train --exp_name 'rfgan_ring' --nepochs '501' --nheads '5' --type 'rfgan' --data_type 'grid'
