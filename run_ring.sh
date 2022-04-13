python -m src.train --exp_name 'gan1_ring' --nepochs '501' --nheads '1' --type 'gan1' --data_type 'ring'
python -m src.train --exp_name 'hgan_ring' --nepochs '501' --nheads '5' --type 'hgan' --data_type 'ring'
python -m src.train --exp_name 'rfgan_ring' --nepochs '501' --nheads '5' --type 'rfgan' --data_type 'ring'
