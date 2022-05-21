SHELL := /bin/bash
.PHONY : all

help:
	cat Makefile

cub_trans:
	CUDA_LAUNCH_BLOCKING=1 python src/train_images_transductive.py --gammaD 10 --gammaG 10 --gammaG_D2 10 --gammaD2 10 \
	--gzsl --manualSeed 3483 --preprocessing --cuda --image_embedding res101 --class_embedding sent \
	--nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0003 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
	--nclass_all 200 --batch_size 64 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 2048 --syn_num 300

awa_trans:
	CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m src/train_images_transductive.py --gammaD 10 --gammaG_D2 10 --gammaD2 10 \
	--gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
	--class_embedding att --nepoch 300 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
	--nclass_all 50 --dataset AWA2 --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
	--lr 0.0007 --classifier_lr 0.001

flo_trans:
	CUDA_LAUNCH_BLOCKING=1 python src/train_images_transductive.py --gammaD 10 --gammaG 10 --gammaG_D2 1 --gammaD2 1 \
	--gzsl --encoded_noise --manualSeed 806 --preprocessing --cuda --image_embedding res101 --class_embedding att \
	--nepoch 700 --ngh 4096 --ndh 4096 --lr 0.0008 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset FLO \
	--nclass_all 102 --batch_size 64 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 2048 --syn_num 1200

sun_trans:
	CUDA_LAUNCH_BLOCKING=1 python src/train_images_transductive.py --gammaD 10 --gammaG 10 --gammaG_D2 10 --gammaD2 10 --gzsl \
	--manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 500 --ngh 4096 \
	--ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.0008 \
	--classifier_lr 0.0005 --syn_num 400 --nclass_all 717

cub_ind:
	CUDA_LAUNCH_BLOCKING=1 python src/train_images_inductive.py --gammaD 10 --gammaG 10 \
	--gzsl --manualSeed 3483 --preprocessing --cuda --image_embedding res101 --class_embedding sent \
	--nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0003 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
	--nclass_all 200 --batch_size 64 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 2048 --syn_num 300

awa_ind:
	CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python src/train_images_inductive.py --gammaD 10 \
	--gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101_finetune \
	--class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
	--nclass_all 50 --dataset AWA2 \
	--batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
	--lr 0.00001 --classifier_lr 0.001

flo_ind:
	CUDA_LAUNCH_BLOCKING=1 python train_images_inductive.py --gammaD 10 --gammaG 10 \
	--gzsl --encoded_noise --manualSeed 806 --preprocessing --cuda --image_embedding res101 --class_embedding att \
	--nepoch 700 --ngh 4096 --ndh 4096 --lr 0.0008 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset FLO \
	--nclass_all 102 --batch_size 64 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 2048 --syn_num 1200

sun_ind:
	CUDA_LAUNCH_BLOCKING=1 python train_images_inductive.py --gammaD 1 --gammaG 1 --gzsl \
	--manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101_finetune --class_embedding att --nepoch 400 --ngh 4096 \
	--ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 \
	--classifier_lr 0.0005 --syn_num 400 --nclass_all 717

lint:
	python -m black ./
	python -m isort --profile black ./