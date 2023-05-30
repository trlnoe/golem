python src/main.py  --seed 1 \
                    --n 100000 \
                    --d 8 \
                    --graph_type ER \
                    --degree 4 \
                    --noise_type gaussian_ev \
                    --equal_variances \
                    --lambda_1 2e-2 \
                    --lambda_2 5.0 \
                    --checkpoint_iter 5000 \
                    --training_path /dataset/Bayesian_Data/ALARM/ALARM_DATA.csv \
                    --gt_path /dataset/Bayesian_Data/ALARM/DAGtrue_ALARM_bi.csv 