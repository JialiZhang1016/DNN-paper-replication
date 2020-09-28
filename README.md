# DNN

**The programming environment is tensorflow2.0 + python3.7**  
**This project is a reproduction of an academic paper, the abstract of the paper is as follows：**  
This work revisits the problem of finding empirical measurement towards overfitting and generalization performance while avoiding the use of testing samples. We average the squared norms of loss gradient per training sample and formulate mean empirical gradient norms (MEGAN). Through our empirical studies, MEGAN has been evidenced to correlate with the generalization gap of deep learning models in a quasi-linear manner. Over the optimization path of deep learning, MEGAN could project the “worst-case” generalization gap when the model overfits the training set. Our theoretical analysis connects MEGAN to the stability of learning dynamics under perturbation and the strength of implicit regularization of stochastic empirical risk minimization (Stochastic ERM). We also conduct several experiments to demonstrate the potential of MEGAN for early stopping and model selection.

## 2020/3/1 update

- [mnist on MLP.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/minist%20on%20MLP.py)
- MLP-3.py
- ![MLP_acc.png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/MLP_acc.png "MLP_acc.png")
- ![MLP_loss.png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/MLP_loss.png "MLP_loss.png")


## 2020/3/10 update

I uploaded 3 py-files and 2 png-files.

- [opt_loop.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/opt_loop.py)  
- [bs_loop.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/bs_loop.py)   
- [opt+bs_loop.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/opt+bs_loop.py)  
- ![Train-Test Accuracy (3-opt,3-bs, 1-lr).png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/Train-Test%20Accuracy%20(3-opt%2C3-bs%2C%201-lr).png "Train-Test Accuracy (3-opt,3-bs, 1-lr).png")
- ![Train-Test Loss (3-opt,3-bs, 1-lr).png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/Train-Test%20Loss%20(3-opt%2C3-bs%2C%201-lr).png "Train-Test Loss (3-opt,3-bs, 1-lr).png")


## 2020/3/13 update

I uploaded 1 py-file and 1 png-file. (solved the problem of legend by creating the list to store the names and results)

- [bs+lr_loop.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/bs+lr_loop.py)  
- ![Training Loss_lr=[0.01,0.001,0.005] bs=[64.128,256].png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/training_loss_lr%3D%5B0.01%2C%200.001%2C%200.005%5D_bs%3D%5B64%2C%20128%2C%20256%5D.png "Training Loss_lr=[0.01,0.001,0.005] bs=[64.128,256].png")


## 2020/4/12 update

I uploaded 1 py-file and 3 png files.

- [MEGAN.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/MEGAN.py)
- ![Megan.png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/Megan.png "Megan.png")
- ![accuracy.png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/accuracy.png "accuracy.png")
- ![loss.png](https://github.com/JialiZhang1016/DNN/blob/master/DNN/loss.png "loss.png")


## 2020/4/19 update

I uploaded 2 .py files.

- [Adaptive Methods.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/Adaptive%20Methods.py)
- [Adaptive Methods_df.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/Adaptive%20Methods_df.py)


## 2020/4/21 update

I uploaded 1 .py file and 4 .png files.
These files are examples for chapter 9 of Nonlinear Optimization in Machine Learning courses.
I added some notes and modified some variable names in this file.
- [Adaptive Methods_optimizer_frame.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/Adaptive%20Methods_optimizer_frame.py)

I uploaded 1 folder.


## 2020/4/27 update

I uploaded 2 .py files.  
- [MEGAN_bs_loop.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/MEGAN_bs_loop.py)
- [MEGAN_lr_loop.py](https://github.com/JialiZhang1016/DNN/blob/master/DNN/MEGAN_lr_loop.py)
 
