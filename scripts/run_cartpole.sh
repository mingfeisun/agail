for seed in $(seq 0 4); 
    do 
        OPENAI_LOG_FORMAT=csv 
        python3 agail_trpo_cartpole.py --loss_percent 0.25 --num_timesteps 50000 --algo agail --seed=$seed &
    done
python3 agail_trpo_cartpole.py --loss_percent 0.25 --num_timesteps 50000 --algo agail --seed=5
echo 'done'
