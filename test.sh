# for i in 101 102 103 105 18 2 22 28 32 41 45 46 55 62 69 80 87 89 90
for i in 1 4 12 13 14 15 16 19 20 25 26 27 29 30 34 35 38 39 43 44 50 51 52 53 59 61 63 68 70 73 76 77 78 79 81 86 91 93
# for i in 46
do
    python tools/demo_track.py image -f exps/example/mot/yolox_s_mix_det.py -c YOLOX_outputs/yolox_s_mix_det/epoch179/latest_ckpt.pth.tar --path datasets/Jerry/train/$i/$i --fp16 --fuse --save_result
        # python tools/demo_track.py image -f exps/example/mot/yolox_x_mix_det.py -c YOLOX_outputs/yolox_x_mix_det/latest_ckpt.pth.tar --path datasets/Jerry/test1/$i/$i --fp16 --fuse --save_result
done
# for i in 103
# do
# python tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py -c YOLOX_outputs/yolox_s_mix_det/epoch179/latest_ckpt.pth.tar --path videos/Jerry103.mp4 --fp16 --fuse --save_result
# done