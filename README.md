# Recaptured Screen Image Demoiréing (AMNet)
This code is the official implementation of TCSVT 2020 paper "Recaptured Screen Image Demoiréing".

Paper:<br>
--------
https://ieeexplore.ieee.org/abstract/document/8972378<br>

Environment:<br>
--------
Windows 8 + Nvidia Titan X GPU <br>
Python (version 3.6.4) + Tensorflow (version 1.10.0) <br>

Network:<br>
-------
 <div align=center><img src="https://github.com/tju-maoyan/AMNet/blob/master/images/Network.png"></div><br>
Fig. 1. The architecture of our AMNet: (a) the generator of our network, comprised of additive (circled by the purple rectangle) and multiplicative (circled by the green rectangle) modules, (b) the ASPP block in the generator network, (c) the multiplicative block in the generator network, and (d) the discriminator of our network. In particular, the “k” represents kernel size, the “n” represents the number of channels, the “s” represents stride size, and the “d” represents the dilation rate. The upsampling layer is realized by 2× nearest neighbor upsampling.<br>

Results:<br>
-------
 <div align=center><img src="https://github.com/tju-maoyan/AMNet/blob/master/images/demoire_exp.png" width="40%" height="40%"></div><br>
Fig. 2. The recaptured screen images (top row), our demoiréing results (the second row), and the corresponding screenshot images (bottom row). Please zoom in the figure for better observation.<br>
<br>

 <div align=center><img src="https://github.com/tju-maoyan/AMNet/blob/master/images/SOTA.png"></div><br>
Fig. 3. Visual quality comparisons for one image captured by Huawei Honor 6X with the screen Philips MWX12201<br>

Download pre-trained model:<br>
--------
`VGG19:` https://pan.baidu.com/s/1YFbPiBYtdIa6ZDmWYJHZJQ (key:l6x1)<br>
`trained model:` https://pan.baidu.com/s/1qvS04gnSSLbqvBCR9K3BAw (key:3kja)<br>

Download dataset:<br>
--------
`Training set:` https://pan.baidu.com/s/1Xn5YygDb9Eg5u5zL3plrsA (key:gpxd)<br>
`Test set:` https://pan.baidu.com/s/1KCZciRYb-MP16u4W1w3X0Q (key:isn6)<br>

Citation<br>
-------
If you find this work useful for your research, please cite:<br>
```
@article{Yue2020Recaptured,
	author = {Yue, Huanjing and Mao, Yan and Liang, Lipu and Xu, Hongteng and Hou, Chunping and Yang, Jingyu},
	year = {2021},
        title = {Recaptured Screen Image Demoir\'eing},
	volume={31},
	number={1},
	pages={49-60},
	journal = {IEEE Transactions on Circuits and Systems for Video Technology},
	doi = {10.1109/TCSVT.2020.2969984}
}
```
