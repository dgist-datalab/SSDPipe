python main_ImageNet.py --freeze True --batch_size 256 --initial_lr 0.001 --print_freq 500 --weight_decay 0.0005 --workers 16 --lr_scheduler_patience 10 --resume ../02_MakeStaleModel/result/model_best.pth --num_class 943 --percent 74