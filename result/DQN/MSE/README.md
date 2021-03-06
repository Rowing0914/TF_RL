# Result Report

All target values are derived from DQN of [Nature paper](https://www.nature.com/articles/nature14236.pdf)

hmm.. i might re-train the agent again.

## Params

|      Item      |              Value              |
| :------------: | :-----------------------------: |
|    loss_fn     |               MSE               |
| grad_clip_flg  |              None               |
|   num_frames   |         10,000,000(10M)         |
| train_interval |                4                |
|  memory_size   |            1,00,000             |
| learning_start |       After 50,000 steps        |
|   sync_freq    |             10,000              |
|   batch_size   |               32                |
|     gamma      |              0.99               |
| target_update  |           hard update           |
| epsilon_start  |               1.0               |
|  epsilon_end   | 0.1 (for evaluation, 0.01 used) |
|  decay_steps   |            1,000,000            |



# Without Target Line

<img src="images/without_target_line/VideoPinball.png" width="70%">
<img src="images/without_target_line/Boxing.png" width="70%">
<img src="images/without_target_line/Breakout.png" width="70%">
<img src="images/without_target_line/StarGunner.png" width="70%">
<img src="images/without_target_line/Robotank.png" width="70%">
<img src="images/without_target_line/Atlantis.png" width="70%">
<img src="images/without_target_line/CrazyClimber.png" width="70%">
<img src="images/without_target_line/Gopher.png" width="70%">
<img src="images/without_target_line/DemonAttack.png" width="70%">
<img src="images/without_target_line/NameThisGame.png" width="70%">
<img src="images/without_target_line/Krull.png" width="70%">
<img src="images/without_target_line/Assault.png" width="70%">
<img src="images/without_target_line/RoadRunner.png" width="70%">
<img src="images/without_target_line/Kangaroo.png" width="70%">
<img src="images/without_target_line/Jamesbond.png" width="70%">
<img src="images/without_target_line/Tennis.png" width="70%">
<img src="images/without_target_line/Pong.png" width="70%">
<img src="images/without_target_line/SpaceInvaders.png" width="70%">
<img src="images/without_target_line/BeamRider.png" width="70%">
<img src="images/without_target_line/Tutankham.png" width="70%">
<img src="images/without_target_line/KungFuMaster.png" width="70%">
<img src="images/without_target_line/Freeway.png" width="70%">
<img src="images/without_target_line/TimePilot.png" width="70%">
<img src="images/without_target_line/Enduro.png" width="70%">
<img src="images/without_target_line/FishingDerby.png" width="70%">
<img src="images/without_target_line/UpNDown.png" width="70%">
<img src="images/without_target_line/IceHockey.png" width="70%">
<img src="images/without_target_line/Hero.png" width="70%">
<img src="images/without_target_line/Asterix.png" width="70%">
<img src="images/without_target_line/BattleZone.png" width="70%">
<img src="images/without_target_line/WizardOfWor.png" width="70%">
<img src="images/without_target_line/ChopperCommand.png" width="70%">
<img src="images/without_target_line/Centipede.png" width="70%">
<img src="images/without_target_line/BankHeist.png" width="70%">
<img src="images/without_target_line/Riverraid.png" width="70%">
<img src="images/without_target_line/Zaxxon.png" width="70%">
<img src="images/without_target_line/Amidar.png" width="70%">
<img src="images/without_target_line/Alien.png" width="70%">
<img src="images/without_target_line/Venture.png" width="70%">
<img src="images/without_target_line/Seaquest.png" width="70%">
<img src="images/without_target_line/DoubleDunk.png" width="70%">
<img src="images/without_target_line/Bowling.png" width="70%">
<img src="images/without_target_line/MsPacman.png" width="70%">
<img src="images/without_target_line/Asteroids.png" width="70%">
<img src="images/without_target_line/Frostbite.png" width="70%">
<img src="images/without_target_line/Gravitar.png" width="70%">
<img src="images/without_target_line/PrivateEye.png" width="70%">
<img src="images/without_target_line/MontezumaRevenge.png" width="70%">


# With Target Line
<img src="images/with_target_line/VideoPinball.png" width="70%">
<img src="images/with_target_line/Boxing.png" width="70%">
<img src="images/with_target_line/Breakout.png" width="70%">
<img src="images/with_target_line/StarGunner.png" width="70%">
<img src="images/with_target_line/Robotank.png" width="70%">
<img src="images/with_target_line/Atlantis.png" width="70%">
<img src="images/with_target_line/CrazyClimber.png" width="70%">
<img src="images/with_target_line/Gopher.png" width="70%">
<img src="images/with_target_line/DemonAttack.png" width="70%">
<img src="images/with_target_line/NameThisGame.png" width="70%">
<img src="images/with_target_line/Krull.png" width="70%">
<img src="images/with_target_line/Assault.png" width="70%">
<img src="images/with_target_line/RoadRunner.png" width="70%">
<img src="images/with_target_line/Kangaroo.png" width="70%">
<img src="images/with_target_line/Jamesbond.png" width="70%">
<img src="images/with_target_line/Tennis.png" width="70%">
<img src="images/with_target_line/Pong.png" width="70%">
<img src="images/with_target_line/SpaceInvaders.png" width="70%">
<img src="images/with_target_line/BeamRider.png" width="70%">
<img src="images/with_target_line/Tutankham.png" width="70%">
<img src="images/with_target_line/KungFuMaster.png" width="70%">
<img src="images/with_target_line/Freeway.png" width="70%">
<img src="images/with_target_line/TimePilot.png" width="70%">
<img src="images/with_target_line/Enduro.png" width="70%">
<img src="images/with_target_line/FishingDerby.png" width="70%">
<img src="images/with_target_line/UpNDown.png" width="70%">
<img src="images/with_target_line/IceHockey.png" width="70%">
<img src="images/with_target_line/Hero.png" width="70%">
<img src="images/with_target_line/Asterix.png" width="70%">
<img src="images/with_target_line/BattleZone.png" width="70%">
<img src="images/with_target_line/WizardOfWor.png" width="70%">
<img src="images/with_target_line/ChopperCommand.png" width="70%">
<img src="images/with_target_line/Centipede.png" width="70%">
<img src="images/with_target_line/BankHeist.png" width="70%">
<img src="images/with_target_line/Riverraid.png" width="70%">
<img src="images/with_target_line/Zaxxon.png" width="70%">
<img src="images/with_target_line/Amidar.png" width="70%">
<img src="images/with_target_line/Alien.png" width="70%">
<img src="images/with_target_line/Venture.png" width="70%">
<img src="images/with_target_line/Seaquest.png" width="70%">
<img src="images/with_target_line/DoubleDunk.png" width="70%">
<img src="images/with_target_line/Bowling.png" width="70%">
<img src="images/with_target_line/MsPacman.png" width="70%">
<img src="images/with_target_line/Asteroids.png" width="70%">
<img src="images/with_target_line/Frostbite.png" width="70%">
<img src="images/with_target_line/Gravitar.png" width="70%">
<img src="images/with_target_line/PrivateEye.png" width="70%">
<img src="images/with_target_line/MontezumaRevenge.png" width="70%">