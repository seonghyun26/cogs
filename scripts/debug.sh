cd ../
current_date=$(TZ="Asia/Seoul" date +%Y%m%d-%H%M%S)

python main.py \
  --config config/debug.yaml \
  --date "$current_date" \
  --device cuda:3