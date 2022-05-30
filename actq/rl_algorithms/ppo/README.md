# Proximal Policy Optimization (PPO)
## Instructions
1. Train the agents - **Atari Env**:
```bash
python train.py --env-name='<env-name>' --cuda (if you have a GPU) --env-type='atari' --lr-decay
```
2. Train the agents - **Mujoco Env** (we also support beta distribution, can use `--dist` flag):
```bash
python train.py --env-name='<env-name>' --cuda (if you have a GPU) --env-type='mujoco' --num-workers=1 --nsteps=2048 --clip=0.2 --batch-size=32 --epoch=10 --lr=3e-4 --ent-coef=0 --total-frames=1000000 --vloss-coef=1 
```
3. Play the demo - Please use the same `--env-type` and `--dist` flag used in the training.
```bash
python demo.py --env-name='<env name>' --env-type='<env type>' --dist='<dist-type>'
```
4. play ppo-mujoco-uniform
```bash
python train.py --env-name=Hopper-v2 --env-type=mujoco --num-workers=1 --nsteps=2048 --clip=0.2 --batch-size=32 --epoch=10 --lr=3e-4 --ent-coef=0 --total-frames=2000000 --vloss-coef=1 --cuda --dist uniform
```
- short version
```bash
python train.py --env-name=Hopper-v2
```
- if we want expert data
```bash
python train.py --env-name=hopper-expert-v2
```
- ppo + k-means and play discrete
```bash
python train.py --env-name=hopper-expert-v2 --cuda --dist ib --updateB kmeans
```
- ppo + Q loss and play discrete
```bash
python train.py --env-name=Hopper-v2 --cuda --dist ib --updateB Q
```
- ppo + actq
```bash
python train.py --env-name=Hopper-v2 --cuda --dist actq --seed 125 --k 5

```
## Results
![](../../figures/05_ppo.png)
