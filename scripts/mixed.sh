cd ../
current_date=$(TZ="Asia/Seoul" date +%Y%m%d-%H%M%S)

python main.py \
  --config config/mixed.yaml \
  --date "$current_date" \
  --device cuda:2