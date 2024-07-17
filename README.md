# Collective variable Guided Sampling 

## Install

This code uses the following libraries
- bgflow
- bgmol


## How to use
Go to the scripts folder, execute any bash file!

For example, the test.sh looks like the following

```bash
cd ../
current_date=$(TZ="Asia/Seoul" date +%Y-%m-%d)

python main.py \
  --config config/basic.yaml \
  --date "$current_date" \
  --device cuda:2
```

main.py has the following arguments

- config: path to config file 
- date: current date (optional)
- device: GPU to use (optional)

Unfortuantely, this code does not support multiple GPUs.