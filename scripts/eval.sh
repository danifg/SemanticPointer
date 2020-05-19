cp ./models/$3/*gold_test$1 ./results/best_gold_test_$3.dag
cp ./models/$3/*pred_test$1 ./results/best_pred_test_$3.dag

cp ./models/$3/*gold_test_two$1 ./results/best_gold_test2_$3.dag
cp ./models/$3/*pred_test_two$1 ./results/best_pred_test2_$3.dag


python ./scripts/deconvert.py ./results/best_gold_test_$3.dag ./results/best_gold_test_$3.sdp
python ./scripts/deconvert.py ./results/best_pred_test_$3.dag ./results/best_pred_test_$3.sdp

python ./scripts/deconvert.py ./results/best_gold_test2_$3.dag ./results/best_gold_test2_$3.sdp
python ./scripts/deconvert.py ./results/best_pred_test2_$3.dag ./results/best_pred_test2_$3.sdp

./scripts/run.sh Scorer ./results/best_gold_test_$3.sdp ./results/best_pred_test_$3.sdp  representation=$2 2> ./results/report
./scripts/run.sh Scorer ./results/best_gold_test2_$3.sdp ./results/best_pred_test2_$3.sdp  representation=$2 2> ./results/report2
echo '## Scores including virtual dependencies to top nodes'
echo 'TEST ID'
cat ./results/report | grep LF: -m 1
cat ./results/report | grep UF: -m 1
echo 'TEST OOD'
cat ./results/report2 | grep LF: -m 1
cat ./results/report2 | grep UF: -m 1
