# DAgger & Behaviour Cloning

### Python version:  **3.6.6**

### Manual

1. Install dependencies on your local environment: `pip install -r requirements.txt`
2. Run `sh collect_experts_demo.sh` to collect the demonstrations of the expert
3. Run `python3.6 behavioural_cloning.py`

   - if you already done this step once, then you don't need to train the model again, so just run `python3.6 behavioural_cloning.py --test`
4. Run `python3.6 DAgger_CartPole.py`

   - if you already done this step once, then you don't need to train the model again, so just run `python3.6 DAgger_CartPole.py --test`

   - If you want to try a randomly behaving agent, then

     `python3.6 DAgger_CartPole.py --random`

### Directory Structure

- `expert_data`: where we store the experts' demo data in Numpy Array format
- `expert_models`: we store the experts' models (DQN and Duelling DQN)
- `videos`: we store the experts' demo video while we collect the demo by `collect_experts_demo.sh`
- `weights`: we store the trained models' weights, e.g., Behavioural Cloning and DAgger
