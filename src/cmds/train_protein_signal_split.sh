python src/main.py  --seed 39 \
                    --n 5836 \
                    --d 11 \
                    --graph_type ER \
                    --degree 4 \
                    --noise_type gaussian_ev \
                    --non_equal_variances \
                    --lambda_1 2e-2 \
                    --lambda_2 5.0 \
                    --checkpoint_iter 5000 \
                    --training_path /dataset/Causal_Protein_Signaling/split_data/data_50.csv \
                    --gt_path /dataset/Causal_Protein_Signaling/split_data/gt.csv