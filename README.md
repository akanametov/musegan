MuseGAN
=========
A Pytorch implementation of MuseGAN

[Check out the generated piano music](https://akanametov.github.io/musegan/)

:star: Star this project on GitHub — it helps!

[MuseGAN](https://arxiv.org/abs/1709.06298) is a generative model which allows to
generate music.

## Table of content

- [Training](#train)
- [Results](#res)
- [License](#license)
- [Links](#links)

## Training 

See [demo](https://github.com/akanametov/CycleGAN/blob/main/demo/demo.ipynb) for more details of training process.
* The models are under `model/__init__.py`.
* Helpfull modules are under `model/modules.py`.
* The model trainer is under `trainer.py`.
### Results
##### `Generators` and `Discriminators` losses

<a><div class="column">
    <img src="images/g_loss_a2o.png" align="center" height="200px" width="300px"/>
    <img src="images/d_loss_a2o.png" align="center" height="200px" width="300px"/>
</div></a>

#### Result on both Generators: from `A2B` and `B2A`

<a><div class="column">
    <img src="images/apple2orange.jpg" align="center" height="400px" width="400px"/>
    <img src="images/orange2apple.jpg" align="center" height="400px" width="400px"/>
</div></a>

## License

This project is licensed under MIT.

## Links

* [MuseGAN](https://arxiv.org/abs/1709.06298)
