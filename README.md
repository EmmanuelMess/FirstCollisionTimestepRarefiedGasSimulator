# First-collision timestep rarefied gas simulator
This simulator computes all possible intersections, then checks which of those possible intersections is soonest to
occur, and does some checks, the simulator then runs for time for enough time for the collision to occur. Once the 
intersection occurs, the whole process runs again.

The idea is that CUDA allows this to go very fast, by computing a lot of data in parallel (all segments and
intersections), this method of simulation is very presice as intesections are computed analitically, not 
in steps.

## Architecture

The general arch is in [architecture.md](https://github.com/EmmanuelMess/FirstCollisionTimestepRarefiedGasSimulator/blob/master/architecture.md).

# Image

<img src="results.gif"/>

# Some refrences and thanks

* [Colliding balls](https://garethrees.org/2009/02/17/physics/): An explanation for the basic idea, but without much implementation info.
* [EasyBMP](https://github.com/izanbf1803/EasyBMP): EasyBMP is an easy to use library to generate BMP images with a simple structure to prototype any project with image generation requirement.
* [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers): Thin, unified, C++-flavored wrappers for the CUDA APIs 

----
<a class="imgpatreon" href="https://www.patreon.com/emmanuelmess" target="_blank">
<img alt="Become a patreon" src="https://user-images.githubusercontent.com/10991116/56376378-07065400-61de-11e9-9583-8ff2148aa41c.png" width=150px></a>
