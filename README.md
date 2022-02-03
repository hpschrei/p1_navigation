[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Description of the Environment

The project environment is a __modified environment__ of the original __Banana Collector Environment__ from the Unity ML-Agents toolkit (located [here](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector)). The modified environment is a custom build by Udacity with the following properties:

* There is a __single agent__ that moves in a planar arena, where the __observations__ are determined by a set of distance-based sensors as well as a couple of intrinsic measurements and the available set of __actions__ of the agent consists of four discrete movement options.
* Additionally, the planar arena is filled with NPCs, which are bananas consisting of two categories: __yellow bananas__, which give the agent a __reward of +1__, and __purple bananas__, which give the agent a __reward of -1__.
* The reinforcement learning task is __episodic__, with a maximum of 300 steps per episode.

#### Agent Observations

The observations that the agent makes in the environment are made in a __37-dimensional continuous space__ corresponding to 35 dimensions of __ray-based perception__ of objects around the agent's forward direction and 2 dimensions of __velocity__.

The 35 dimensions of __ray perception__ are further broken down in the following way:

* There are 7 rays emanating from the agent at the following __angles__: [20, 90, 160, 45, 135, 70, 110] where 90 is right in front of the agent.
* Each ray is 5-dimensional and is projected onto the environment, with each entry serving as an __encoding for an object in the environment__. If a ray encounters one of four detectable objects (yellow banana, wall, blue banana, agent), the value at that position in the array is set to 1.
* The final entry in the 5-dimensional array is a __distance measure__ which is the fraction of the ray length at which an object was encountered. Thus, each ray is a vector with the representation [Yellow Banana, Wall, Blue Banana, Agent, Distance]. The array [0, 1, 1, 0, 0, 0.2] for example represents a ray that hit a blue banana 20% of the distance along the ray with a wall behind it.

The __velocity__ of the agent is 2-dimensional: left/right velocity (usually near 0) and forward/backward velocity (0 to 11.2).

![rays](C:\Users\hansp\Dropbox\Courses\Deep_Reinforcement_Learning_[Udacity]\deep-reinforcement-learning\p1_navigation\images\rays.png)

#### Agent actions

The actions that the agent can take consist of 4 discrete actions that indicate the direction of movement of the agent in the plane. The actions are indexed in the following manner:

* __Action 0__: Move forward.
* __Action 1__: Move backward.
* __Action 2__: Turn left.
* __Action 3__: Turn right.

#### Environment dynamics and rewards

The agent is initially placed at a random point in the arena plane, which is limited by walls. When the agent comes in contact with a banana (either yellow or blue), the agent receives the respective reward (+1 if yellow banana, -1 if blue banana). The exercise is considered solved once the agent can consistently get an average reward of +13 over 100 episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 
