batch_size=64
num_workers=42
img_sz=512
max_epochs=15
lr=1e-3
model_name="resnext50_32x4d_ssl"
loss_fn="label_smoothing"
aug_p=0.5
gpus=2


python lightning.py --fold_id=0 --gpus=$gpus --loss_fn=$loss_fn  --aug_p=$aug_p --batch_size=$batch_size --lr=$lr --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name

# for i in {0..4}
# do
#    python lightning.py --fold_id=$i --loss_fn=$loss_fn --aug_p=$aug_p --batch_size=$batch_size --lr=$lr --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name
# done
