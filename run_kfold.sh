batch_size=48
num_workers=42
img_sz=512
max_epochs=5
lr=0.001
model_name="resnext50_32x4d_ssl"


for i in {0..4}
do
   python lightning.py --fold_id=$i --batch_size=$batch_size --lr=$lr --num_workers=$num_workers --img_sz=$img_sz --max_epochs=$max_epochs --model_name=$model_name
done
