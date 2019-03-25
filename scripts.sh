###
###        Train
###
python PyTorch_main_train_Rain100.py --input_path ./data/rain \
        --gt_path ./data/clean \
        --save_dir ./model

###
###        Test
###
python PyTorch_main_test_Rain100.py --input_path_test ./Data/Rain100H_Test/rain/X2 --gt_path_test ./Data/Rain100H_Test/norain \
        --model_dir ./model --output_dir ./experiments
