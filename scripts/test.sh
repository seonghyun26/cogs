cd ../
current_date=$(TZ="Asia/Seoul" date +%Y-%m-%d)

python main.py \
  --config config/basic.yaml \
  --date "$current_date" \
  --device cuda:2