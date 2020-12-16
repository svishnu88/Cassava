batch_size=32
num_workers=42
img_sz=512
max_epochs=5
model_name="resnext50_32x4d_ssl"


python lightning.py --fold_id=0 --batch_size=$batch_size --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name

python lightning.py --fold_id=1 --batch_size=$batch_size --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name

python lightning.py --fold_id=2 --batch_size=$batch_size --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name

python lightning.py --fold_id=3 --batch_size=$batch_size --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name

python lightning.py --fold_id=4 --batch_size=$batch_size --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name
