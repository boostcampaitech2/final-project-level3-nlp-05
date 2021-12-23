# Make pyarrow dataset

```sh
python make_parquet.py
```



# Train
```sh
python train.py --do_train --do_eval --do_predict --use_wandb --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --num_train_epochs 20 --eval_steps 16384 --gradient_accumulation_steps 128 --seed 42 --learning_rate 1e-5 --weight_decay 0.01 --repetition_penalty 2.0 --no_repeat_ngram_size 3
```


# Inference
```sh
python inference.py --model_dir ./saved --date 20211217
```
- 요약문 생성에 쓰일 json 파일과 생성 결과가 저장된 json 파일을 덮어쓰려면 `--overwrite` 추가


# Demo

Please refer to `demo/demo.ipynb`.