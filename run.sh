# detect
python main.py --config_yaml experiments/gid_sd_60.yaml --train.mode pretrain
python main.py --config_yaml experiments/gid_sd_80.yaml --train.mode pretrain
python main.py --config_yaml experiments/gid_sd_90.yaml --train.mode pretrain

python main.py --config_yaml experiments/gid_cd_60.yaml --train.mode pretrain
python main.py --config_yaml experiments/gid_cd_80.yaml --train.mode pretrain
python main.py --config_yaml experiments/gid_cd_90.yaml --train.mode pretrain

python main.py --config_yaml experiments/gid_md_60.yaml --train.mode pretrain
python main.py --config_yaml experiments/gid_md_80.yaml --train.mode pretrain
python main.py --config_yaml experiments/gid_md_90.yaml --train.mode pretrain


# discover train
python main.py --config_yaml experiments/gid_sd_60.yaml --train.mode discover --discover logs/gid-sd-60/GID-SD
python main.py --config_yaml experiments/gid_sd_80.yaml --train.mode discover --discover logs/gid-sd-80/GID-SD
python main.py --config_yaml experiments/gid_sd_90.yaml --train.mode discover --discover logs/gid-sd-90/GID-SD

python main.py --config_yaml experiments/gid_cd_60.yaml --train.mode discover --discover logs/gid-cd-60/GID-CD
python main.py --config_yaml experiments/gid_cd_80.yaml --train.mode discover --discover logs/gid-cd-80/GID-CD
python main.py --config_yaml experiments/gid_cd_90.yaml --train.mode discover --discover logs/gid-cd-90/GID-CD

python main.py --config_yaml experiments/gid_md_60.yaml --train.mode discover --discover logs/gid-md-60/GID-MD
python main.py --config_yaml experiments/gid_md_80.yaml --train.mode discover --discover logs/gid-md-80/GID-MD
python main.py --config_yaml experiments/gid_md_90.yaml --train.mode discover --discover logs/gid-md-90/GID-MD

# discover test
python main.py --config_yaml experiments/gid_sd_60.yaml --train.mode discover --test logs/gid-sd-60/GID-SD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-SD-60
python main.py --config_yaml experiments/gid_sd_60.yaml --train.mode discover --test logs/gid-sd-60/GID-SD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-SD-60
python main.py --config_yaml experiments/gid_sd_80.yaml --train.mode discover --test logs/gid-sd-80/GID-SD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-SD-80
python main.py --config_yaml experiments/gid_sd_80.yaml --train.mode discover --test logs/gid-sd-80/GID-SD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-SD-80
python main.py --config_yaml experiments/gid_sd_90.yaml --train.mode discover --test logs/gid-sd-90/GID-SD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-SD-90
python main.py --config_yaml experiments/gid_sd_90.yaml --train.mode discover --test logs/gid-sd-90/GID-SD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-SD-90
python main.py --config_yaml experiments/gid_cd_60.yaml --train.mode discover --test logs/gid-cd-60/GID-CD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-CD-60
python main.py --config_yaml experiments/gid_cd_60.yaml --train.mode discover --test logs/gid-cd-60/GID-CD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-CD-60
python main.py --config_yaml experiments/gid_cd_80.yaml --train.mode discover --test logs/gid-cd-80/GID-CD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-CD-80
python main.py --config_yaml experiments/gid_cd_80.yaml --train.mode discover --test logs/gid-cd-80/GID-CD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-CD-80
python main.py --config_yaml experiments/gid_cd_90.yaml --train.mode discover --test logs/gid-cd-90/GID-CD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-CD-90
python main.py --config_yaml experiments/gid_cd_90.yaml --train.mode discover --test logs/gid-cd-90/GID-CD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-CD-90
python main.py --config_yaml experiments/gid_md_60.yaml --train.mode discover --test logs/gid-md-60/GID-MD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-MD-60
python main.py --config_yaml experiments/gid_md_60.yaml --train.mode discover --test logs/gid-md-60/GID-MD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-MD-60
python main.py --config_yaml experiments/gid_md_80.yaml --train.mode discover --test logs/gid-md-80/GID-MD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-MD-80
python main.py --config_yaml experiments/gid_md_80.yaml --train.mode discover --test logs/gid-md-80/GID-MD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-MD-80
python main.py --config_yaml experiments/gid_md_90.yaml --train.mode discover --test logs/gid-md-90/GID-MD --dataset.path datasets/TextClassification/GID_IND_TEST/GID-MD-90
python main.py --config_yaml experiments/gid_md_90.yaml --train.mode discover --test logs/gid-md-90/GID-MD --dataset.path datasets/TextClassification/GID_OOD_TEST/GID-MD-90
