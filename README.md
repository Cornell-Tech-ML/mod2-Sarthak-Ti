[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py

# task 2_5:
## simple dataset
parameters:
* lr=0.1
* epochs=500
* hidden_layer_size=2
* accuracy=100%
* time_per_epoch=0.041s
* num_points=50

![image](https://github.com/user-attachments/assets/9d012b9e-0d3b-43b8-9ec1-01848c08c688)

```console
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 36.277377010551945, correct: 23
Epoch: 20/500, loss: 34.527091259349156, correct: 23
Epoch: 30/500, loss: 33.85299368515139, correct: 29
Epoch: 40/500, loss: 33.427198927789796, correct: 43
Epoch: 50/500, loss: 33.05863416496171, correct: 41
Epoch: 60/500, loss: 32.636254458931035, correct: 41
Epoch: 70/500, loss: 32.1044570279743, correct: 43
Epoch: 80/500, loss: 31.421093733770352, correct: 43
Epoch: 90/500, loss: 30.51715728430997, correct: 47
Epoch: 100/500, loss: 29.324642691424923, correct: 50
Epoch: 110/500, loss: 27.80522289288489, correct: 50
Epoch: 120/500, loss: 26.00685305508253, correct: 49
Epoch: 130/500, loss: 24.08996074983407, correct: 49
Epoch: 140/500, loss: 22.019154842908776, correct: 49
Epoch: 150/500, loss: 20.032986538480873, correct: 49
Epoch: 160/500, loss: 18.19928678347761, correct: 49
Epoch: 170/500, loss: 16.53024421063915, correct: 49
Epoch: 180/500, loss: 15.120423126646017, correct: 49
Epoch: 190/500, loss: 13.92759623546035, correct: 49
Epoch: 200/500, loss: 12.90120020529976, correct: 49
Epoch: 210/500, loss: 11.991204362187311, correct: 49
Epoch: 220/500, loss: 11.169808239795618, correct: 50
Epoch: 230/500, loss: 10.447339981497098, correct: 50
Epoch: 240/500, loss: 9.794560410695468, correct: 50
Epoch: 250/500, loss: 9.202343092844513, correct: 50
Epoch: 260/500, loss: 8.663909041615897, correct: 50
Epoch: 270/500, loss: 8.180553164543092, correct: 50
Epoch: 280/500, loss: 7.747640158648691, correct: 50
Epoch: 290/500, loss: 7.351354590286042, correct: 50
Epoch: 300/500, loss: 6.9872944888598685, correct: 50
Epoch: 310/500, loss: 6.651770485773617, correct: 50
Epoch: 320/500, loss: 6.341623471773919, correct: 50
Epoch: 330/500, loss: 6.0595342454800765, correct: 50
Epoch: 340/500, loss: 5.802231851879403, correct: 50
Epoch: 350/500, loss: 5.568237706009259, correct: 50
Epoch: 360/500, loss: 5.351336360767528, correct: 50
Epoch: 370/500, loss: 5.147859821202965, correct: 50
Epoch: 380/500, loss: 4.956560068431211, correct: 50
Epoch: 390/500, loss: 4.776364056345968, correct: 50
Epoch: 400/500, loss: 4.606338270665696, correct: 50
Epoch: 410/500, loss: 4.445665585008615, correct: 50
Epoch: 420/500, loss: 4.293626233557169, correct: 50
Epoch: 430/500, loss: 4.149582186168382, correct: 50
Epoch: 440/500, loss: 4.012964329050002, correct: 50
Epoch: 450/500, loss: 3.883261952082782, correct: 50
Epoch: 460/500, loss: 3.7600141269276808, correct: 50
Epoch: 470/500, loss: 3.64280263039364, correct: 50
Epoch: 480/500, loss: 3.531246127036235, correct: 50
Epoch: 490/500, loss: 3.4249953751637117, correct: 50
Epoch: 500/500, loss: 3.323729262654433, correct: 50
```

## diag dataset
parameters:
* lr=0.5
* epochs=500
* hidden_layer_size=4
* accuracy=100%
* time_per_epoch=0.083s
* num_points=50

![{EB5CAA9E-9429-46F9-B84D-0EC92DEEB816}](https://github.com/user-attachments/assets/c7ce17cf-e8e5-4f00-b6e4-f5b125c8a57e)

```console
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 22.272410668760074, correct: 42
Epoch: 20/500, loss: 21.84094372558751, correct: 42
Epoch: 30/500, loss: 21.66752911924863, correct: 42
Epoch: 40/500, loss: 21.449786508250966, correct: 42
Epoch: 50/500, loss: 21.069759004188388, correct: 42
Epoch: 60/500, loss: 20.379697927862527, correct: 42
Epoch: 70/500, loss: 19.20717758449665, correct: 42
Epoch: 80/500, loss: 17.540761124570174, correct: 42
Epoch: 90/500, loss: 15.403465762565057, correct: 42
Epoch: 100/500, loss: 13.139299728596082, correct: 42
Epoch: 110/500, loss: 11.063494641177641, correct: 42
Epoch: 120/500, loss: 9.312742398631668, correct: 42
Epoch: 130/500, loss: 7.824398774271011, correct: 46
Epoch: 140/500, loss: 6.731497883069457, correct: 49
Epoch: 150/500, loss: 5.629349977236235, correct: 49
Epoch: 160/500, loss: 4.902354191042115, correct: 49
Epoch: 170/500, loss: 4.274558444303941, correct: 49
Epoch: 180/500, loss: 3.7874776314181413, correct: 49
Epoch: 190/500, loss: 3.3888046965563166, correct: 49
Epoch: 200/500, loss: 3.074023989484533, correct: 49
Epoch: 210/500, loss: 2.8162356985143298, correct: 49
Epoch: 220/500, loss: 2.6033285646083604, correct: 49
Epoch: 230/500, loss: 2.4231891363712244, correct: 49
Epoch: 240/500, loss: 2.27507124375667, correct: 49
Epoch: 250/500, loss: 2.1489276196354283, correct: 49
Epoch: 260/500, loss: 2.040046552357576, correct: 49
Epoch: 270/500, loss: 1.9448803859822246, correct: 49
Epoch: 280/500, loss: 1.8614510908578046, correct: 49
Epoch: 290/500, loss: 1.7887832808765947, correct: 49
Epoch: 300/500, loss: 1.7233263374146552, correct: 49
Epoch: 310/500, loss: 1.6649852337962692, correct: 49
Epoch: 320/500, loss: 1.6116263982509302, correct: 49
Epoch: 330/500, loss: 1.5623409386023193, correct: 49
Epoch: 340/500, loss: 1.5164383167261921, correct: 49
Epoch: 350/500, loss: 1.4733871178723268, correct: 49
Epoch: 360/500, loss: 1.4327733262492046, correct: 49
Epoch: 370/500, loss: 1.3942705677376885, correct: 49
Epoch: 380/500, loss: 1.3576186299822017, correct: 49
Epoch: 390/500, loss: 1.3225326298810087, correct: 50
Epoch: 400/500, loss: 1.2889243690521053, correct: 50
Epoch: 410/500, loss: 1.2566595107150118, correct: 50
Epoch: 420/500, loss: 1.2256198394308522, correct: 50
Epoch: 430/500, loss: 1.1957059691211294, correct: 50
Epoch: 440/500, loss: 1.1668335664631582, correct: 50
Epoch: 450/500, loss: 1.1389304219263903, correct: 50
Epoch: 460/500, loss: 1.1119341610635023, correct: 50
Epoch: 470/500, loss: 1.0857629686802235, correct: 50
Epoch: 480/500, loss: 1.0604025390039047, correct: 50
Epoch: 490/500, loss: 1.0358117898674382, correct: 50
Epoch: 500/500, loss: 1.0119520954197911, correct: 50
```

## split dataset
parameters:
* lr=0.1
* epochs=1000
* hidden_layer_size=8
* accuracy=98%
* time_per_epoch=0.209s
* num_points=50

![image](https://github.com/user-attachments/assets/9365b19f-a5d8-411c-97c6-87937fce2552)

```console
Epoch: 0/1000, loss: 0, correct: 0
Epoch: 10/1000, loss: 32.08552334220576, correct: 32
Epoch: 20/1000, loss: 31.427322709105283, correct: 34
Epoch: 30/1000, loss: 30.821933867968493, correct: 34
Epoch: 40/1000, loss: 30.454962186167332, correct: 34
Epoch: 50/1000, loss: 30.051907945237687, correct: 34
Epoch: 60/1000, loss: 29.64500084105941, correct: 35
Epoch: 70/1000, loss: 29.2026581101891, correct: 35
Epoch: 80/1000, loss: 28.71440511627001, correct: 35
Epoch: 90/1000, loss: 28.19500636299414, correct: 35
Epoch: 100/1000, loss: 27.644841185116082, correct: 35
Epoch: 110/1000, loss: 27.070426198946322, correct: 35
Epoch: 120/1000, loss: 26.432837433162902, correct: 35
Epoch: 130/1000, loss: 25.764106824069458, correct: 36
Epoch: 140/1000, loss: 25.060906844314548, correct: 40
Epoch: 150/1000, loss: 24.320865594787573, correct: 43
Epoch: 160/1000, loss: 23.548148657555828, correct: 44
Epoch: 170/1000, loss: 22.753355874829207, correct: 44
Epoch: 180/1000, loss: 21.92964677612981, correct: 45
Epoch: 190/1000, loss: 21.086289042933878, correct: 45
Epoch: 200/1000, loss: 20.227781932844135, correct: 45
Epoch: 210/1000, loss: 19.32401653734312, correct: 46
Epoch: 220/1000, loss: 18.392439204310463, correct: 46
Epoch: 230/1000, loss: 17.383841768619785, correct: 46
Epoch: 240/1000, loss: 16.52846273352088, correct: 46
Epoch: 250/1000, loss: 15.757759240793582, correct: 46
Epoch: 260/1000, loss: 15.035703289289566, correct: 46
Epoch: 270/1000, loss: 14.359185487416095, correct: 46
Epoch: 280/1000, loss: 13.722893295632119, correct: 47
Epoch: 290/1000, loss: 13.120792565120855, correct: 47
Epoch: 300/1000, loss: 12.53455810606931, correct: 47
Epoch: 310/1000, loss: 12.022776439636237, correct: 47
Epoch: 320/1000, loss: 11.548725352390752, correct: 47
Epoch: 330/1000, loss: 11.11034856141552, correct: 47
Epoch: 340/1000, loss: 10.70513485633763, correct: 47
Epoch: 350/1000, loss: 10.330506458690818, correct: 47
Epoch: 360/1000, loss: 9.984560473017815, correct: 47
Epoch: 370/1000, loss: 9.664823465352232, correct: 47
Epoch: 380/1000, loss: 9.369254797648528, correct: 47
Epoch: 390/1000, loss: 9.095800903685204, correct: 47
Epoch: 400/1000, loss: 8.842807254369847, correct: 47
Epoch: 410/1000, loss: 8.605314030525019, correct: 47
Epoch: 420/1000, loss: 8.384583815593086, correct: 47
Epoch: 430/1000, loss: 8.179577009630412, correct: 48
Epoch: 440/1000, loss: 7.9889464896799876, correct: 49
Epoch: 450/1000, loss: 7.811706335787472, correct: 49
Epoch: 460/1000, loss: 7.646448383353533, correct: 49
Epoch: 470/1000, loss: 7.49229540871376, correct: 49
Epoch: 480/1000, loss: 7.348155913223681, correct: 49
Epoch: 490/1000, loss: 7.211393874743591, correct: 49
Epoch: 500/1000, loss: 7.082878368899727, correct: 49
Epoch: 510/1000, loss: 6.961738911680517, correct: 49
Epoch: 520/1000, loss: 6.837091783958783, correct: 49
Epoch: 530/1000, loss: 6.720285435857065, correct: 49
Epoch: 540/1000, loss: 6.616679802177031, correct: 49
Epoch: 550/1000, loss: 6.520383664157271, correct: 49
Epoch: 560/1000, loss: 6.431262454784389, correct: 49
Epoch: 570/1000, loss: 6.346987247759842, correct: 49
Epoch: 580/1000, loss: 6.266918991949215, correct: 49
Epoch: 590/1000, loss: 6.190628097221657, correct: 49
Epoch: 600/1000, loss: 6.117997735873618, correct: 49
Epoch: 610/1000, loss: 6.048722968472945, correct: 49
Epoch: 620/1000, loss: 5.982615760842001, correct: 49
Epoch: 630/1000, loss: 5.919388901026802, correct: 49
Epoch: 640/1000, loss: 5.858876980788575, correct: 49
Epoch: 650/1000, loss: 5.800912782576659, correct: 49
Epoch: 660/1000, loss: 5.7453507102116035, correct: 49
Epoch: 670/1000, loss: 5.691899695258833, correct: 49
Epoch: 680/1000, loss: 5.640472177430936, correct: 49
Epoch: 690/1000, loss: 5.591169396430598, correct: 49
Epoch: 700/1000, loss: 5.543690541236856, correct: 49
Epoch: 710/1000, loss: 5.497876628495809, correct: 49
Epoch: 720/1000, loss: 5.453658223880283, correct: 49
Epoch: 730/1000, loss: 5.410968966973472, correct: 49
Epoch: 740/1000, loss: 5.369680973385952, correct: 49
Epoch: 750/1000, loss: 5.329704849770121, correct: 49
Epoch: 760/1000, loss: 5.290886155566712, correct: 49
Epoch: 770/1000, loss: 5.253272584838814, correct: 49
Epoch: 780/1000, loss: 5.216906024386446, correct: 49
Epoch: 790/1000, loss: 5.181440610006257, correct: 49
Epoch: 800/1000, loss: 5.147043438089998, correct: 49
Epoch: 810/1000, loss: 5.113575213263311, correct: 49
Epoch: 820/1000, loss: 5.081012435749419, correct: 49
Epoch: 830/1000, loss: 5.049285013337459, correct: 49
Epoch: 840/1000, loss: 5.018448039773204, correct: 49
Epoch: 850/1000, loss: 4.988247971195266, correct: 49
Epoch: 860/1000, loss: 4.958915415316003, correct: 49
Epoch: 870/1000, loss: 4.930301557688052, correct: 49
Epoch: 880/1000, loss: 4.9022596315078735, correct: 49
Epoch: 890/1000, loss: 4.874867418261826, correct: 49
Epoch: 900/1000, loss: 4.8480965036056025, correct: 49
Epoch: 910/1000, loss: 4.821985761462337, correct: 49
Epoch: 920/1000, loss: 4.79645550944466, correct: 49
Epoch: 930/1000, loss: 4.771457225781084, correct: 49
Epoch: 940/1000, loss: 4.746969302884128, correct: 49
Epoch: 950/1000, loss: 4.72297122110075, correct: 49
Epoch: 960/1000, loss: 4.69944951465436, correct: 49
Epoch: 970/1000, loss: 4.6764031195105265, correct: 49
Epoch: 980/1000, loss: 4.653773534622255, correct: 49
Epoch: 990/1000, loss: 4.631604059340102, correct: 49
Epoch: 1000/1000, loss: 4.609840634375287, correct: 49
```

## xor dataset
parameters:
* lr=0.1
* epochs=1000
* hidden_layer_size=16
* accuracy=100%
* num_points=50
* time_per_epoch=0.646s

![image](https://github.com/user-attachments/assets/58f85f5c-e083-4652-9c75-cd8467367436)

```console
Epoch: 0/1000, loss: 0, correct: 0
Epoch: 10/1000, loss: 34.499925994411406, correct: 25
Epoch: 20/1000, loss: 33.74495714189479, correct: 22
Epoch: 30/1000, loss: 33.29592458344291, correct: 24
Epoch: 40/1000, loss: 32.83901265358182, correct: 26
Epoch: 50/1000, loss: 32.36409478628276, correct: 30
Epoch: 60/1000, loss: 31.923622596218568, correct: 34
Epoch: 70/1000, loss: 31.511505203774398, correct: 37
Epoch: 80/1000, loss: 31.047543610693587, correct: 38
Epoch: 90/1000, loss: 30.612692553830016, correct: 39
Epoch: 100/1000, loss: 30.15023846364753, correct: 39
Epoch: 110/1000, loss: 29.668651064234105, correct: 39
Epoch: 120/1000, loss: 29.148678847650856, correct: 40
Epoch: 130/1000, loss: 28.61582749127105, correct: 39
Epoch: 140/1000, loss: 28.075604121632978, correct: 39
Epoch: 150/1000, loss: 26.919004602190395, correct: 39
Epoch: 160/1000, loss: 26.06824480223739, correct: 39
Epoch: 170/1000, loss: 25.34347569481509, correct: 40
Epoch: 180/1000, loss: 24.67697456386485, correct: 41
Epoch: 190/1000, loss: 24.0452017502077, correct: 40
Epoch: 200/1000, loss: 23.359326390337397, correct: 40
Epoch: 210/1000, loss: 22.710011496966448, correct: 40
Epoch: 220/1000, loss: 22.071943587327716, correct: 40
Epoch: 230/1000, loss: 21.508776083491338, correct: 40
Epoch: 240/1000, loss: 20.945614398124324, correct: 40
Epoch: 250/1000, loss: 20.3998848819175, correct: 41
Epoch: 260/1000, loss: 19.88842916113236, correct: 41
Epoch: 270/1000, loss: 19.406477740812733, correct: 41
Epoch: 280/1000, loss: 18.94491044098856, correct: 41
Epoch: 290/1000, loss: 18.501141789332117, correct: 41
Epoch: 300/1000, loss: 18.073034065086127, correct: 41
Epoch: 310/1000, loss: 17.667232471955593, correct: 41
Epoch: 320/1000, loss: 17.275002009562066, correct: 41
Epoch: 330/1000, loss: 16.909743182890043, correct: 42
Epoch: 340/1000, loss: 16.55655283765242, correct: 42
Epoch: 350/1000, loss: 16.216384560386082, correct: 43
Epoch: 360/1000, loss: 15.883643090712962, correct: 43
Epoch: 370/1000, loss: 15.560312068203553, correct: 43
Epoch: 380/1000, loss: 15.239832620940991, correct: 44
Epoch: 390/1000, loss: 14.932198080458857, correct: 44
Epoch: 400/1000, loss: 14.629078704010675, correct: 44
Epoch: 410/1000, loss: 14.328686506876435, correct: 44
Epoch: 420/1000, loss: 14.025465239789112, correct: 44
Epoch: 430/1000, loss: 13.723790915671206, correct: 45
Epoch: 440/1000, loss: 13.423790371704357, correct: 45
Epoch: 450/1000, loss: 13.12779436186108, correct: 45
Epoch: 460/1000, loss: 12.834786938175094, correct: 45
Epoch: 470/1000, loss: 12.542524480737399, correct: 45
Epoch: 480/1000, loss: 12.256142914487908, correct: 47
Epoch: 490/1000, loss: 11.977525166653876, correct: 47
Epoch: 500/1000, loss: 11.705831252467565, correct: 47
Epoch: 510/1000, loss: 11.434665101035483, correct: 47
Epoch: 520/1000, loss: 11.175616975008316, correct: 47
Epoch: 530/1000, loss: 10.925984510220356, correct: 47
Epoch: 540/1000, loss: 10.679227484342531, correct: 47
Epoch: 550/1000, loss: 10.442443328977687, correct: 47
Epoch: 560/1000, loss: 10.21672593624404, correct: 47
Epoch: 570/1000, loss: 9.99764630048615, correct: 47
Epoch: 580/1000, loss: 9.782613896277832, correct: 47
Epoch: 590/1000, loss: 9.577255944428012, correct: 47
Epoch: 600/1000, loss: 9.382020856534714, correct: 47
Epoch: 610/1000, loss: 9.19563167145722, correct: 47
Epoch: 620/1000, loss: 9.023326549092747, correct: 47
Epoch: 630/1000, loss: 8.856794007222675, correct: 48
Epoch: 640/1000, loss: 8.69269230837471, correct: 48
Epoch: 650/1000, loss: 8.53800251212963, correct: 48
Epoch: 660/1000, loss: 8.383722753453874, correct: 48
Epoch: 670/1000, loss: 8.244289843207758, correct: 48
Epoch: 680/1000, loss: 8.101435044306088, correct: 48
Epoch: 690/1000, loss: 7.976233005829902, correct: 48
Epoch: 700/1000, loss: 7.841616544304265, correct: 48
Epoch: 710/1000, loss: 7.729183465577818, correct: 48
Epoch: 720/1000, loss: 7.614220556167864, correct: 48
Epoch: 730/1000, loss: 7.512607203145536, correct: 48
Epoch: 740/1000, loss: 7.413239817540149, correct: 48
Epoch: 750/1000, loss: 7.312898068373077, correct: 48
Epoch: 760/1000, loss: 7.2216100000058505, correct: 48
Epoch: 770/1000, loss: 7.131739274387962, correct: 48
Epoch: 780/1000, loss: 7.049676477132239, correct: 48
Epoch: 790/1000, loss: 6.968944935148339, correct: 48
Epoch: 800/1000, loss: 6.890544989054634, correct: 48
Epoch: 810/1000, loss: 6.815941234402155, correct: 48
Epoch: 820/1000, loss: 6.7455977290511, correct: 48
Epoch: 830/1000, loss: 6.678067350253568, correct: 48
Epoch: 840/1000, loss: 6.604836782597232, correct: 48
Epoch: 850/1000, loss: 6.550775335358252, correct: 48
Epoch: 860/1000, loss: 6.480397553897371, correct: 48
Epoch: 870/1000, loss: 6.425562659508591, correct: 48
Epoch: 880/1000, loss: 6.371152609328891, correct: 48
Epoch: 890/1000, loss: 6.316237274750619, correct: 48
Epoch: 900/1000, loss: 6.2601778528360335, correct: 47
Epoch: 910/1000, loss: 6.210471762238512, correct: 47
Epoch: 920/1000, loss: 6.165074960632226, correct: 47
Epoch: 930/1000, loss: 6.115649995333382, correct: 47
Epoch: 940/1000, loss: 6.06838663359138, correct: 47
Epoch: 950/1000, loss: 6.024705709831898, correct: 47
Epoch: 960/1000, loss: 5.979785672244132, correct: 47
Epoch: 970/1000, loss: 5.943717821512608, correct: 47
Epoch: 980/1000, loss: 5.906472449357485, correct: 47
Epoch: 990/1000, loss: 5.866749588734957, correct: 47
Epoch: 1000/1000, loss: 5.825129095886816, correct: 47
Epoch: 0/1000, loss: 0, correct: 0
Epoch: 10/1000, loss: 31.15885433631989, correct: 34
Epoch: 20/1000, loss: 29.168248330168247, correct: 45
Epoch: 30/1000, loss: 27.852570298335923, correct: 45
Epoch: 40/1000, loss: 26.74964460435816, correct: 46
Epoch: 50/1000, loss: 25.66749541606077, correct: 46
Epoch: 60/1000, loss: 24.592946590191524, correct: 46
Epoch: 70/1000, loss: 23.504637866015777, correct: 46
Epoch: 80/1000, loss: 22.3379389299052, correct: 46
Epoch: 90/1000, loss: 21.253939377976398, correct: 46
Epoch: 100/1000, loss: 20.207272837418035, correct: 46
Epoch: 110/1000, loss: 19.202513177696368, correct: 47
Epoch: 120/1000, loss: 18.176297986359653, correct: 48
Epoch: 130/1000, loss: 16.887391926056615, correct: 48
Epoch: 140/1000, loss: 15.661999534037719, correct: 48
Epoch: 150/1000, loss: 14.626120055369299, correct: 49
Epoch: 160/1000, loss: 13.818592517962218, correct: 49
Epoch: 170/1000, loss: 13.151431918232964, correct: 49
Epoch: 180/1000, loss: 12.547926740755099, correct: 49
Epoch: 190/1000, loss: 11.990947771404613, correct: 49
Epoch: 200/1000, loss: 11.48220617088983, correct: 49
Epoch: 210/1000, loss: 11.01393006947037, correct: 50
Epoch: 220/1000, loss: 10.544958887821158, correct: 50
Epoch: 230/1000, loss: 10.071560023596117, correct: 50
Epoch: 240/1000, loss: 9.661851968299187, correct: 50
Epoch: 250/1000, loss: 9.294410740357037, correct: 50
Epoch: 260/1000, loss: 8.953929914604515, correct: 50
Epoch: 270/1000, loss: 8.574950124082148, correct: 50
Epoch: 280/1000, loss: 8.139312249721408, correct: 50
Epoch: 290/1000, loss: 7.7798644381179, correct: 50
Epoch: 300/1000, loss: 7.512610045291273, correct: 50
Epoch: 310/1000, loss: 7.27015008365543, correct: 50
Epoch: 320/1000, loss: 7.056304005558696, correct: 50
Epoch: 330/1000, loss: 6.857571046997345, correct: 50
Epoch: 340/1000, loss: 6.669180030536226, correct: 50
Epoch: 350/1000, loss: 6.493288509932261, correct: 50
Epoch: 360/1000, loss: 6.327739305060571, correct: 50
Epoch: 370/1000, loss: 6.169368972250866, correct: 50
Epoch: 380/1000, loss: 6.019956129579375, correct: 50
Epoch: 390/1000, loss: 5.877713908159215, correct: 50
Epoch: 400/1000, loss: 5.741440531657282, correct: 50
Epoch: 410/1000, loss: 5.610883410669363, correct: 50
Epoch: 420/1000, loss: 5.48651411432369, correct: 50
Epoch: 430/1000, loss: 5.36749329551934, correct: 50
Epoch: 440/1000, loss: 5.252544243156859, correct: 50
Epoch: 450/1000, loss: 5.143091175287479, correct: 50
Epoch: 460/1000, loss: 5.032826688561207, correct: 50
Epoch: 470/1000, loss: 4.917147466738136, correct: 50
Epoch: 480/1000, loss: 4.804732363585584, correct: 50
Epoch: 490/1000, loss: 4.7091344711194205, correct: 50
Epoch: 500/1000, loss: 4.617687408012635, correct: 50
Epoch: 510/1000, loss: 4.53058433288246, correct: 50
Epoch: 520/1000, loss: 4.447616431579765, correct: 50
Epoch: 530/1000, loss: 4.367731342464886, correct: 50
Epoch: 540/1000, loss: 4.290640298255674, correct: 50
Epoch: 550/1000, loss: 4.216028737660512, correct: 50
Epoch: 560/1000, loss: 4.144155604091928, correct: 50
Epoch: 570/1000, loss: 4.0750060900210805, correct: 50
Epoch: 580/1000, loss: 4.007269120443527, correct: 50
Epoch: 590/1000, loss: 3.942590220759018, correct: 50
Epoch: 600/1000, loss: 3.879264057101551, correct: 50
Epoch: 610/1000, loss: 3.817618475661988, correct: 50
Epoch: 620/1000, loss: 3.7577443816452503, correct: 50
Epoch: 630/1000, loss: 3.700189680578533, correct: 50
Epoch: 640/1000, loss: 3.6432601849106243, correct: 50
Epoch: 650/1000, loss: 3.5886427091316895, correct: 50
Epoch: 660/1000, loss: 3.5357424954157373, correct: 50
Epoch: 670/1000, loss: 3.4828982282504795, correct: 50
Epoch: 680/1000, loss: 3.4323259709264415, correct: 50
Epoch: 690/1000, loss: 3.3826899075580434, correct: 50
Epoch: 700/1000, loss: 3.3342437177624284, correct: 50
Epoch: 710/1000, loss: 3.2873390980432027, correct: 50
Epoch: 720/1000, loss: 3.2416662248973296, correct: 50
Epoch: 730/1000, loss: 3.197213452039884, correct: 50
Epoch: 740/1000, loss: 3.1532824188501114, correct: 50
Epoch: 750/1000, loss: 3.1111771985226686, correct: 50
Epoch: 760/1000, loss: 3.0692095709852496, correct: 50
Epoch: 770/1000, loss: 3.02844716601033, correct: 50
Epoch: 780/1000, loss: 2.988757788961123, correct: 50
Epoch: 790/1000, loss: 2.9499298336336537, correct: 50
Epoch: 800/1000, loss: 2.912552256707149, correct: 50
Epoch: 810/1000, loss: 2.8747126768324676, correct: 50
Epoch: 820/1000, loss: 2.8381716642494985, correct: 50
Epoch: 830/1000, loss: 2.8023561181020455, correct: 50
Epoch: 840/1000, loss: 2.7673571568598687, correct: 50
Epoch: 850/1000, loss: 2.7331214940308097, correct: 50
Epoch: 860/1000, loss: 2.699206120178405, correct: 50
Epoch: 870/1000, loss: 2.6664060524317854, correct: 50
Epoch: 880/1000, loss: 2.6347073764088043, correct: 50
Epoch: 890/1000, loss: 2.6030941971370067, correct: 50
Epoch: 900/1000, loss: 2.5720115820440292, correct: 50
Epoch: 910/1000, loss: 2.541881190201985, correct: 50
Epoch: 920/1000, loss: 2.5118255421714046, correct: 50
Epoch: 930/1000, loss: 2.4830663190179516, correct: 50
Epoch: 940/1000, loss: 2.454081011327006, correct: 50
Epoch: 950/1000, loss: 2.4265301568414084, correct: 50
Epoch: 960/1000, loss: 2.3987804691428334, correct: 50
Epoch: 970/1000, loss: 2.372199161143713, correct: 50
Epoch: 980/1000, loss: 2.344988311118902, correct: 50
Epoch: 990/1000, loss: 2.3191772456164323, correct: 50
Epoch: 1000/1000, loss: 2.2933612682123368, correct: 50
```
