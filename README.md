# 2048-api
A 2048 game api for training supervised learning (imitation learning) agents

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

#代码结构：
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.I added TrainAgent and TestAgent classes.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    * [`CNN.py`](game2048/CNN.py): a convolution nueral network for trainning.
    * [`one_hotboard.py`](game2048/one_hotboard.py): transfer board to onehot board.
    * [`calculatedata.py`](game2048/calculatedata.py): use some properties of the board let train faster.
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* ['train.py'](train.py): used to train CNN

#代码运行：将git网址下的代码clone到本地后在pycharm中新建project并打开，直接运行evaluate，等到控制台输出结束命令时在my2048-api文件夹下的EE369.log文件中可以得到将结果。或者在my2048-api中打开终端，输入chmod a+x evaluate.py，回车后再输入python evaluate.py即可。权重文件是model2048.pkl，我在weight中存放，同时也已经提前拷贝了一份放进了代码文件目录中，以直接运行代码。linux条件下webapp用firefox无法打开，请助教老师留意。

#LICENSE:我的LICENSE放在了my2048-api文件夹内，并非没有
